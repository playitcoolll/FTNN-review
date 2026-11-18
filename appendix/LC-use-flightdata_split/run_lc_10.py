"""LC：对out_1000–out_1009分别做一次最小二乘线性校准，共10次。

运行：python run_lc_10.py
检查数据、源模型和10次最小二乘求解，不写结果：python run_lc_10.py --check-only
续跑：python run_lc_10.py --resume
模型 Ps_LC = a * Ps_source + b；源网络冻结，只用700条训练数据求解a、b。
无需学习率、迭代轮数或新初始化种子；十次差异来自十个源域预训练模型。
航迹固定0、2.5、6、12、25，排除4和15；验证200、测试100、LHS600。
每次保存校准系数、校准前后指标、逐点预测、完整模型和归一化参数。
结果目录results_10sources_lc；all_run_metrics.csv为10次×4集合的40行。
各次RMSE/MAE先独立计算再求平均，不对预测值平均后计算指标。
依赖tensorflow 2.15.1、numpy、pandas、openpyxl。
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent / 'data'
SOURCE_ROOT=PROJECT/'MSFS747-MODEL'/'out_pcnn_5D'
SOURCE_SEEDS=list(range(1000,1010))
# 飞行航迹白名单，所有划分共用，明确不包含4和15。
TRACKS=['0','2.5','6','12','25']
LAYERS=[5,64,64,64,1]
# 仅用于原类创建占位网络，随即全部被预训练权重覆盖，不产生新的实验随机性。
RANDOM_SEED=1234
ALGORITHM_ORIGIN='Residual LINE-MODEL/train_residualline_fix_5D_fd.py'
EMBEDDED_ALGORITHM_SHA256='fdcbd1322e290c6e3dde26af86883a21b1c3292f6215a2b0642d5a16f4c36fa1'

def fit_linear_calibration(source_prediction, target):
    """最小化 mean((target - (a * source_prediction + b)) ** 2)。"""
    source_prediction = np.asarray(source_prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if source_prediction.size != target.size or target.size < 2:
        raise ValueError("源模型预测与目标标签必须等长，且至少包含两个样本")
    if not np.isfinite(source_prediction).all() or not np.isfinite(target).all():
        raise ValueError("校准数据包含 NaN 或 Inf")

    # 每个样本只有一个解释变量，即源模型的标量输出；不对五维输入另做回归。
    design_matrix = np.column_stack([source_prediction, np.ones_like(source_prediction)])
    parameters, _, rank, singular_values = np.linalg.lstsq(
        design_matrix, target, rcond=None
    )
    if rank < 2:
        raise ValueError("源模型训练集预测没有足够变化，无法唯一识别校准参数 a、b")
    if not np.isfinite(parameters).all():
        raise FloatingPointError("线性校准参数出现 NaN 或 Inf")
    return parameters, int(rank), singular_values


class SourceOutputCalibration:

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        checkpoint_path: Path,
    ):
        self.x_data = np.asarray(x_data, dtype=np.float32)
        self.y_data = np.asarray(y_data, dtype=np.float64).reshape(-1, 1)
        self.lb = np.asarray(lb, dtype=np.float32)
        self.ub = np.asarray(ub, dtype=np.float32)
        if (
            self.x_data.ndim != 2
            or self.x_data.shape[1] != 5
            or len(self.x_data) != len(self.y_data)
            or len(self.x_data) < 2
            or not np.isfinite(self.x_data).all()
            or not np.isfinite(self.y_data).all()
        ):
            raise ValueError("校准训练数据必须为有限的 N×5 输入和对应标签，且 N >= 2")
        if (
            self.lb.shape != (5,)
            or self.ub.shape != (5,)
            or not np.isfinite(self.lb).all()
            or not np.isfinite(self.ub).all()
            or np.any(self.ub <= self.lb)
        ):
            raise ValueError("源域归一化边界必须为五维有限数组，且 ub > lb")

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(RANDOM_SEED)
            with tf.variable_scope("source_model"):
                self.weights, self.biases = self.initialize_nn(LAYERS)
            self.source_variables = self.weights + self.biases
            with tf.variable_scope("calibration"):
                # 初始 a=1、b=0，校准前的输出与源模型相同。
                self.a = tf.Variable(1.0, dtype=tf.float64, name="a")
                self.b = tf.Variable(0.0, dtype=tf.float64, name="b")
            self.calibration_variables = [self.a, self.b]
            self.variables = self.source_variables + self.calibration_variables
            self.x_data_tf = tf.placeholder(tf.float32, shape=[None, 5])
            self.y_data_tf = tf.placeholder(tf.float64, shape=[None, 1])
            self.source_pred = tf.stop_gradient(self.neural_net(self.x_data_tf))
            self.y_pred = tf.add(
                self.a * tf.cast(self.source_pred, tf.float64), self.b,
                name="calibrated_prediction",
            )
            self.loss_data = tf.reduce_mean(tf.square(self.y_data_tf - self.y_pred))
            self.loss = self.loss_data
            self.parameters_tf = tf.placeholder(tf.float64, shape=[2])
            self.assign_parameters = tf.group(
                tf.assign(self.a, self.parameters_tf[0]),
                tf.assign(self.b, self.parameters_tf[1]),
            )

            # 源模型按 W0、b0、W1、b1 顺序生成 Variable、Variable_1、...。
            checkpoint_variables = {}
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                weight_name = "Variable" if index == 0 else f"Variable_{2 * index}"
                checkpoint_variables[weight_name] = weight
                checkpoint_variables[f"Variable_{2 * index + 1}"] = bias
            self.source_saver = tf.train.Saver(var_list=checkpoint_variables)
            self.saver = tf.train.Saver(var_list=self.variables)
            initializer = tf.global_variables_initializer()

        self.feed_dict = {
            self.x_data_tf: self.x_data,
            self.y_data_tf: self.y_data,
        }
        self.history: list[list[float]] = []
        self.fit_information = {}
        self.sess = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            ),
        )
        try:
            self.sess.run(initializer)
            self.source_saver.restore(self.sess, str(checkpoint_path))
        except Exception:
            self.sess.close()
            raise
        self.source_parameters_before = self.sess.run(self.source_variables)

    def xavier_init(self, size, name):
        stddev = np.sqrt(2.0 / (size[0] + size[1]))
        return tf.Variable(
            tf.truncated_normal(size, stddev=stddev), trainable=False, name=name
        )

    def initialize_nn(self, layers):
        weights = []
        biases = []
        for index in range(len(layers) - 1):
            weights.append(
                self.xavier_init([layers[index], layers[index + 1]], f"weight_{index}")
            )
            biases.append(
                tf.Variable(
                    tf.zeros([1, layers[index + 1]], dtype=tf.float32),
                    trainable=False, name=f"bias_{index}",
                )
            )
        return weights, biases

    def neural_net(self, x):
        hidden = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for index in range(len(self.weights) - 1):
            hidden = tf.tanh(
                tf.add(tf.matmul(hidden, self.weights[index]), self.biases[index])
            )
        return tf.add(tf.matmul(hidden, self.weights[-1]), self.biases[-1])

    def evaluate_losses(self):
        return self.sess.run([self.loss, self.loss_data], feed_dict=self.feed_dict)

    def record(self, iteration: int) -> None:
        values = self.evaluate_losses()
        a, b = self.sess.run(self.calibration_variables)
        if not np.isfinite([*values, a, b]).all():
            raise FloatingPointError("线性校准损失或参数出现 NaN 或 Inf")
        row = [iteration, float(values[0]), float(values[1]), float(a), float(b)]
        self.history.append(row)
        print(
            "Calibration step %d: total=%.6e, data=%.6e, a=%.8f, b=%.8f"
            % tuple(row)
        )

    def source_parameters_unchanged(self) -> bool:
        return all(
            np.array_equal(before, after)
            for before, after in zip(
                self.source_parameters_before, self.sess.run(self.source_variables)
            )
        )

    def train(self):
        self.history = []
        self.record(0)
        # 仅训练集进入最小二乘求解；验证集和测试集仅用于后续评价。
        source_prediction = self.predict_source(self.x_data)
        parameters, rank, singular_values = fit_linear_calibration(
            source_prediction, self.y_data
        )
        self.sess.run(self.assign_parameters, {self.parameters_tf: parameters})
        self.fit_information = {
            "design_matrix_rank": rank,
            "singular_values": singular_values.tolist(),
            "condition_number": float(singular_values[0] / singular_values[-1]),
        }
        self.record(1)
        if not self.source_parameters_unchanged():
            raise RuntimeError("源域神经网络参数发生变化，冻结检查失败")
        return np.asarray(self.history, dtype=float)

    def predict_source(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(self.source_pred, {self.x_data_tf: x})

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(self.y_pred, {self.x_data_tf: x})

def frame_to_model_data(frame):
    mach=frame['mach'].to_numpy(dtype=np.float32)
    height=frame['h'].to_numpy(dtype=np.float32)
    mass=frame['mass'].to_numpy(dtype=np.float32)
    n1=frame['N1'].to_numpy(dtype=np.float32)
    temperature=frame['dtem'].to_numpy(dtype=np.float32)
    x=np.column_stack([mach,height/10000.0,mass/100000.0,n1/100.0,
                       (temperature+273.15)/273.15]).astype(np.float32)
    y=frame['Ps'].to_numpy(dtype=np.float32).reshape(-1,1)
    return x,y

alg=SimpleNamespace(tf=tf,LAYERS=LAYERS,frame_to_model_data=frame_to_model_data)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def experiment_settings(value):
    """Compare run settings/data while ignoring historical source-code metadata."""
    code_keys = {'algorithms', 'algorithm_hashes', 'script_sha256',
                 'sampling_script_sha256', 'algorithm_sha256',
                 'original_algorithm_sha256', 'evaluator_sha256',
                 'embedded_algorithm_sha256'}
    if isinstance(value, dict):
        return {k: experiment_settings(v) for k, v in value.items() if k not in code_keys}
    if isinstance(value, list):
        return [experiment_settings(v) for v in value]
    return value


def same_settings(left, right):
    return experiment_settings(left) == experiment_settings(right)



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def metric(y,p):
    y,p=np.asarray(y,dtype=np.float64).ravel(),np.asarray(p,dtype=np.float64).ravel()
    require(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid predictions')
    error=p-y
    sst=np.sum((y-y.mean())**2)
    require(sst>0,'Cannot compute R2 for constant target')
    return dict(n=len(y),mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))),r2=float(1-np.sum(error**2)/sst))

def load_data(args):
    frames, arrays, files = {},{},[]
    for split,n in [('train',140),('validation',40),('test',20)]:
        parts=[]
        for track in TRACKS:
            found=list(args.data_root.glob(f'out_20_{track}_90_70_*_{split}.xlsx'))
            require(len(found)==1,f'Expected one file for track={track}, split={split}: {found}')
            path=found[0]
            f=pd.read_excel(path,engine='openpyxl').round(2)
            require(len(f)==n,f'Unexpected sample count: {path}')
            require(np.isfinite(f[['t','mach','h','mass','N1','dtem','V','Ps']].to_numpy()).all(),f'Invalid values: {path}')
            require((f.V>0).all(),f'Invalid velocity: {path}')
            f['track']=track
            f['source_file']=path.name
            f['source_excel_row']=np.arange(len(f))+2
            parts.append(f)
            files.append(path)
        frames[split]=pd.concat(parts,ignore_index=True)
        arrays[split]=alg.frame_to_model_data(frames[split])
    ids={s:set(zip(f.track,f.t)) for s,f in frames.items()}
    require(all(len(ids[s])==len(frames[s]) for s in frames),'Duplicate trajectory/time records')
    require(not(ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test']),'Overlapping splits')
    raw=pd.read_excel(args.lhs_file,sheet_name='Results',engine='openpyxl')
    fields=['Mach','Hp_m','mass_kg','N1_pct','deltaT_degC']
    valid=(raw.Success==1)&np.isfinite(raw[fields+['Ps_mps']].to_numpy()).all(axis=1)
    audit=raw.copy()
    audit['source_excel_row']=np.arange(len(raw))+2
    audit['included_in_metrics']=valid
    lhs=audit.loc[valid].copy().reset_index(drop=True)
    require(len(lhs)>1,'No usable LHS references')
    mapped=lhs.rename(columns=dict(zip(fields,['mach','h','mass','N1','dtem'])))
    mapped['Ps']=lhs.Ps_mps
    x,_=alg.frame_to_model_data(mapped)
    frames['lhs_target']=lhs
    arrays['lhs_target']=(x,lhs.Ps_mps.to_numpy(dtype=np.float64).reshape(-1,1))
    files.append(args.lhs_file)
    return frames,arrays,audit,files

def load_sources(args):
    sources=[]
    for seed in SOURCE_SEEDS:
        directory=args.source_root/f'out_{seed}'
        norm=directory/'normalization_params.npz'
        configuration=directory/'run_config.json'
        ckpt=directory/'pcnn_model.ckpt'
        cfg=json.loads(configuration.read_text(encoding='utf-8-sig'))
        require(cfg['initialization_seed']==seed and cfg['layers']==alg.LAYERS,f'Wrong source configuration: {directory}')
        with np.load(norm,allow_pickle=False) as z:
            lb,ub=z['lb'].astype(np.float32),z['ub'].astype(np.float32)
            require(lb.shape==(5,) and ub.shape==(5,) and np.isfinite([lb,ub]).all() and (ub>lb).all(),f'Invalid bounds: {norm}')
            for key,value in dict(height_scale=10000.,mass_scale=100000.,dt_scale=100.,
                                  celsius_to_kelvin=273.15,standard_temperature_k=273.15).items():
                require(np.isclose(float(z[key]),value),f'Incompatible scaling: {key}')
            require(z['input_order'].tolist()==['mach','height_scaled','mass_scaled','N1_scaled','temperature_scaled'],'Wrong input order')
        parts=sorted(directory.glob('pcnn_model.ckpt.data-*'))
        require(parts and ckpt.with_suffix('.ckpt.index').is_file(),f'Missing checkpoint: {ckpt}')
        # Read via an ASCII temporary path for Windows TensorFlow compatibility.
        with tempfile.TemporaryDirectory(prefix='ftnn_source_') as tmp:
            for p in [ckpt.with_suffix('.ckpt.index')]+parts:
                shutil.copy2(p,Path(tmp)/p.name)
            reader=alg.tf.train.load_checkpoint(str(Path(tmp)/ckpt.name))
            weights={}
            for i,(a,b) in enumerate(zip(alg.LAYERS[:-1],alg.LAYERS[1:])):
                for k,shape in [(2*i,(a,b)),(2*i+1,(1,b))]:
                    name='Variable' if k==0 else f'Variable_{k}'
                    value=reader.get_tensor(name)
                    require(value.shape==shape and np.isfinite(value).all(),f'Invalid tensor {name}: {ckpt}')
                    weights[name]=value
            del reader
        paths=[configuration,norm,ckpt.with_suffix('.ckpt.index')]+parts
        source=dict(seed=seed,directory=directory,lb=lb,ub=ub,weights=weights,
                    config=cfg,files={str(p.resolve()):sha(p) for p in paths},
                    weight_hash=hashlib.sha256(b''.join(weights[k].tobytes() for k in sorted(weights))).hexdigest())
        sources.append(source)
    require(len({s['weight_hash'] for s in sources})==10,'The ten source weight sets are not distinct')
    require(len({s['config']['split_seed'] for s in sources})==1,'Source split seeds differ')
    return sources

def fresh_model(source,arrays):
    # ASCII临时路径避免Windows TensorFlow读取中文路径检查点的问题。
    with tempfile.TemporaryDirectory(prefix='lc_source_') as tmp:
        for path in source['directory'].glob('pcnn_model.ckpt.*'):
            if path.suffix=='.index' or '.data-' in path.name:
                shutil.copy2(path,Path(tmp)/path.name)
        model=SourceOutputCalibration(*arrays['train'],source['lb'],source['ub'],Path(tmp)/'pcnn_model.ckpt')
    try:
        for i,(w,b) in enumerate(zip(model.weights,model.biases)):
            key='Variable' if i==0 else f'Variable_{2*i}'
            require(np.array_equal(model.sess.run(w),source['weights'][key]),'Source weights not restored exactly')
            require(np.array_equal(model.sess.run(b),source['weights'][f'Variable_{2*i+1}']),'Source biases not restored exactly')
        return model
    except Exception:
        model.sess.close(); raise


def fit_and_check(model,arrays):
    history=model.train()
    require(model.source_parameters_unchanged(),'Source parameters changed')
    before,after=history[:,2]
    require(after<=before+1e-10*max(1.,before),'OLS training MSE increased')
    p=model.predict_source(arrays['train'][0]).astype(np.float64).ravel()
    pred=model.predict(arrays['train'][0]).ravel()
    design=np.column_stack([p,np.ones_like(p)])
    residual=pred-arrays['train'][1].ravel().astype(np.float64)
    scale=max(1.,np.linalg.norm(design)*np.linalg.norm(residual))
    require(np.linalg.norm(design.T@residual)<=1e-10*scale,'OLS normal equations check failed')
    return history


def summarize(frame):
    records=[]
    for split,group in frame.groupby('split',sort=False):
        require(len(group)==10 and set(group.source_seed)==set(SOURCE_SEEDS),'Incomplete group')
        row=dict(split=split,seed_count=10,sample_count=int(group.n.iloc[0]))
        for key in ['mse','rmse','mae','r2','initial_mse','initial_rmse','initial_mae','initial_r2']:
            values=group[key].to_numpy(dtype=float)
            row[key+'_mean']=float(values.mean()); row[key+'_std']=float(values.std(ddof=1))
        records.append(row)
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--data-root',type=Path,default=PROJECT/'inputfile'/'flightdata'/'flightdata_split')
    p.add_argument('--lhs-file',type=Path,default=PROJECT/'inputfile'/'lhs_target'/'LHS_target_trim_Ps_20261028_203140_904.xlsx')
    p.add_argument('--output',type=Path,default=ROOT/'results_10sources_lc')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    frames,arrays,audit,paths=load_data(args); sources=load_sources(args)
    manifest=dict(model='LC',source_seeds=SOURCE_SEEDS,source_frozen=True,
                  calibration='Ps_LC = a * Ps_source + b',solver='numpy.linalg.lstsq; rcond=None',
                  fit_split='train',tuned_hyperparameters=[],regularization='none',
                  flight_tracks=TRACKS,flight_round_decimals=2,lhs_rounding='none',
                  counts={s:len(f) for s,f in frames.items()},lhs_excluded=int((~audit.included_in_metrics).sum()),
                  inputs={str(x.resolve()):sha(x) for x in paths},
                  sources=[dict(seed=s['seed'],weights_sha256=s['weight_hash'],files=s['files']) for s in sources],
                  original_algorithm=ALGORITHM_ORIGIN,embedded_algorithm_sha256=EMBEDDED_ALGORITHM_SHA256,
                  script_sha256=sha(__file__),python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,pandas=pd.__version__)
    print('Data:',manifest['counts'],'Source seeds:',SOURCE_SEEDS,flush=True)
    if args.check_only:
        for source in sources:
            model=fresh_model(source,arrays)
            try:
                fit_and_check(model,arrays)
                for x,y in arrays.values(): metric(y,model.predict(x))
                print(f"CHECK OK: source {source['seed']}; source frozen, OLS optimality checked",flush=True)
            finally: model.sess.close()
        print('CHECK PASSED: 10 calibrations; no result files written.',flush=True)
        return
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        require(args.resume,'Output exists: choose another --output or use --resume')
        require(same_settings(json.loads((out/'experiment_config.json').read_text(encoding='utf-8')), manifest),'Resume refused: settings, sources, inputs, or code changed')
    out.mkdir(parents=True,exist_ok=True); write_json(out/'experiment_config.json',manifest)
    for split,f in frames.items(): f.to_csv(out/f'{split}_data_used.csv',index=False,encoding='utf-8-sig')
    audit.to_csv(out/'lhs_input_audit.csv',index=False,encoding='utf-8-sig')
    all_rows=[]; parameters=[]
    for source in sources:
        seed=source['seed']; run=out/f'source_{seed}'; marker=run/'completed.json'
        if marker.exists():
            saved=json.loads(marker.read_text(encoding='utf-8'))
            require(all((run/f).is_file() and sha(run/f)==h for f,h in saved['files'].items()),f'Completed files changed: {run}')
            all_rows.extend(pd.read_csv(run/'metrics.csv').to_dict('records'))
            parameters.append(json.loads((run/'calibration_parameters.json').read_text(encoding='utf-8')))
            pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
            print(f'SKIP completed: source {seed}',flush=True); continue
        run.mkdir(parents=True,exist_ok=True); model=fresh_model(source,arrays)
        try:
            start=time.perf_counter(); history=fit_and_check(model,arrays); elapsed=time.perf_counter()-start
            a,b=[float(v) for v in model.sess.run(model.calibration_variables)]
            info=dict(source_seed=seed,a=a,b=b,**model.fit_information)
            write_json(run/'calibration_parameters.json',info)
            write_json(run/'run_config.json',dict(source_seed=seed,source_weights_sha256=source['weight_hash'],
                       formula='a * Ps_source + b',fit_split='train',solver='numpy.linalg.lstsq; rcond=None',
                       elapsed_fit_seconds=elapsed,source_frozen=True,source_directory=str(source['directory'])))
            pd.DataFrame(history,columns=['step','total_loss','train_mse','a','b']).to_csv(run/'history.csv',index=False)
            (run/'run.log').write_text(f'source_seed={seed}\na={a:.17g}\nb={b:.17g}\nsource_frozen=True\nfit_split=train\n',encoding='utf-8')
            np.savez(run/'normalization_params.npz',lb=source['lb'],ub=source['ub'],
                     input_order=np.asarray(['mach','height_scaled','mass_scaled','N1_scaled','temperature_scaled']),
                     height_scale=10000.,mass_scale=100000.,dt_scale=100.,celsius_to_kelvin=273.15,standard_temperature_k=273.15)
            rows=[]
            for split,(x,y) in arrays.items():
                before=model.predict_source(x).ravel(); pred=model.predict(x).ravel()
                row=dict(source_seed=seed,split=split,**metric(y,pred),a=a,b=b,elapsed_fit_seconds=elapsed)
                row.update({f'initial_{k}':v for k,v in metric(y,before).items() if k!='n'}); rows.append(row)
                f=frames[split].copy(); f['ps_true']=y.ravel(); f['ps_pred_before']=before; f['ps_pred']=pred
                f['ps_correction']=pred-before.astype(np.float64)
                f['error_pred_minus_true']=pred-y.ravel().astype(np.float64); f['absolute_error']=abs(f.error_pred_minus_true)
                f.to_csv(run/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False,encoding='utf-8-sig')
            model.saver.save(model.sess,str(run/'model.ckpt'),write_meta_graph=False)
            write_json(marker,dict(status='complete',files={f.name:sha(f) for f in run.iterdir() if f.is_file() and f.name!='completed.json'}))
            all_rows.extend(rows); parameters.append(info)
            pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
        except Exception as exc:
            (run/'error.txt').write_text(repr(exc),encoding='utf-8'); raise
        finally: model.sess.close()
    require(len(all_rows)==40,'Expected 10 sources x 4 datasets')
    summarize(pd.DataFrame(all_rows)).to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(parameters).to_csv(out/'calibration_parameters_all.csv',index=False,encoding='utf-8-sig')
    print(f'FINISHED: 10 LC models, 40 metric rows. Results: {out}',flush=True)

if __name__=='__main__':
    main()
