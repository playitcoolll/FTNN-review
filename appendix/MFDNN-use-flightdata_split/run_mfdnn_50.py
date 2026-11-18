"""MFDNN：1000–1009 十个初始化种子 × 五个学习率，共50次联合训练。

运行：python run_mfdnn_50.py
检查50种初始化，不训练、不写结果：python run_mfdnn_50.py --check-only
中断后跳过完整完成的运行：python run_mfdnn_50.py --resume
默认3000轮全批量Adam；每次从头初始化三个子网络，alpha初值0.5并参与训练。
损失=MSE_low+MSE_high，无正则/早停，不读取预训练权重。
高保真固定flightdata_split，仅0、2.5、6、12、25，不包含4、15。
低保真使用MSFS747-MODEL相同Excel、筛选条件及固定1234划分。
同一种子的五个学习率使用相同初值；不同种子不改变任何数据划分。
保存50个模型、逐轮损失、逐点预测、指标与数据审计；不自动选择学习率。
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
import random
import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def train_test_split(indices, test_size, random_state, shuffle=True):
    """Reproduce sklearn ShuffleSplit: ceil(n * test_size), test indices first.

    Uses the same NumPy RandomState permutation, without an extra dependency.
    """
    if not shuffle:
        raise ValueError('Only the source algorithm shuffle=True split is supported')
    n_test=int(np.ceil(len(indices)*test_size))
    order=np.random.RandomState(random_state).permutation(len(indices))
    return indices[order[n_test:]],indices[order[:n_test]]


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'data'
ALGORITHM_ORIGIN = 'MFDNN-JOINT-MODEL/train_mfdnn_joint_5D_fd.py'  # Provenance only.
SOURCE_ROOT = PROJECT/'MSFS747-MODEL'
RATES = [1e-4,5e-4,1e-3,5e-3,1e-2]
INITIALIZATION_SEEDS = list(range(1000,1010))
# 飞行航迹白名单：明确排除4和15，三个固定划分均使用此列表。
TRACKS = ['0','2.5','6','12','25']
SPLIT_SEED = 1234
HEIGHT_SCALE=10000.0
MASS_SCALE=100000.0
DT_SCALE=100.0
CELSIUS_TO_KELVIN=273.15
STANDARD_TEMPERATURE_K=273.15
SOURCE_LAYERS=[5,64,64,64,1]
LINEAR_LAYERS=[6,10,1]
NONLINEAR_LAYERS=[6,64,64,64,1]
PS_SCALE=100.0
LAMBDA_LOW=1.0
LAMBDA_HIGH=1.0
MAX_EPOCHS=3000
PRINT_EVERY=500

EMBEDDED_ALGORITHM_SHA256 = '634630a099c048c523bfe7c492205aa75408bee9b69ecb6ef9ca2174075ff168'

class MFDNN:

    def __init__(
        self,
        x_low: np.ndarray,
        y_low: np.ndarray,
        x_data: np.ndarray,
        y_data: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        fixed_alpha: float | None = None,
        seed: int = 1000,
        learning_rate: float = 1e-3,
    ):
        self.x_low = np.asarray(x_low, dtype=np.float32)
        self.y_low = np.asarray(y_low, dtype=np.float32).reshape(-1, 1)
        self.x_data = np.asarray(x_data, dtype=np.float32)
        self.y_data = np.asarray(y_data, dtype=np.float32).reshape(-1, 1)
        self.lb = np.asarray(lb, dtype=np.float32)
        self.ub = np.asarray(ub, dtype=np.float32)
        for x, y in ((self.x_low, self.y_low), (self.x_data, self.y_data)):
            if (
                x.ndim != 2 or x.shape[1] != 5 or len(x) != len(y)
                or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all()
            ):
                raise ValueError("训练数据必须为有限的 N×5 输入及对应标签，且 N > 0")
        if (
            self.lb.shape != (5,) or self.ub.shape != (5,)
            or not np.isfinite(self.lb).all() or not np.isfinite(self.ub).all()
            or np.any(self.ub <= self.lb)
        ):
            raise ValueError("归一化边界必须为五维有限数组，且 ub > lb")
        for name, value in (
            ("learning_rate", learning_rate), ("LAMBDA_LOW", LAMBDA_LOW),
            ("LAMBDA_HIGH", LAMBDA_HIGH), ("PS_SCALE", PS_SCALE),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为有限正数")
        if MAX_EPOCHS < 0 or PRINT_EVERY < 1:
            raise ValueError("MAX_EPOCHS 必须非负，PRINT_EVERY 必须大于等于 1")
        if fixed_alpha is not None and (
            not np.isfinite(fixed_alpha) or not 0.0 <= fixed_alpha <= 1.0
        ):
            raise ValueError("fixed_alpha 必须为 [0, 1] 内的有限数值")
        self.fixed_alpha = fixed_alpha

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            with tf.variable_scope("low_fidelity"):
                self.low_weights, self.low_biases = self.initialize_nn(SOURCE_LAYERS)
            with tf.variable_scope("high_fidelity_linear"):
                # 原文采用一层 10 神经元隐藏层；每一层均不使用激活函数。
                self.linear_weights, self.linear_biases = self.initialize_nn(LINEAR_LAYERS)
            with tf.variable_scope("high_fidelity_nonlinear"):
                self.nonlinear_weights, self.nonlinear_biases = self.initialize_nn(
                    NONLINEAR_LAYERS
                )
            with tf.variable_scope("multifidelity"):
                if fixed_alpha is None:
                    self.alpha_logit = tf.Variable(0.0, dtype=tf.float32, name="alpha_logit")
                    self.alpha = tf.sigmoid(self.alpha_logit, name="alpha")
                    mixing_variables = [self.alpha_logit]
                    fixed_mixing_variables = []
                else:
                    self.alpha_logit = None
                    self.alpha = tf.Variable(
                        fixed_alpha, dtype=tf.float32, trainable=False, name="alpha_fixed"
                    )
                    mixing_variables = []
                    fixed_mixing_variables = [self.alpha]
                self.ps_scale = tf.Variable(
                    PS_SCALE, dtype=tf.float32, trainable=False, name="ps_scale"
                )

            self.low_variables = self.low_weights + self.low_biases
            self.high_variables = (
                self.linear_weights + self.linear_biases
                + self.nonlinear_weights + self.nonlinear_biases + mixing_variables
            )
            self.trainable_variables = self.low_variables + self.high_variables
            self.variables = self.trainable_variables + fixed_mixing_variables + [self.ps_scale]
            self.x_low_tf = tf.placeholder(tf.float32, shape=[None, 5], name="x_low")
            self.y_low_tf = tf.placeholder(tf.float32, shape=[None, 1], name="y_low")
            self.x_data_tf = tf.placeholder(tf.float32, shape=[None, 5], name="x_high")
            self.y_data_tf = tf.placeholder(tf.float32, shape=[None, 1], name="y_high")
            self.low_pred = self.neural_net(
                self.normalize(self.x_low_tf), self.low_weights, self.low_biases
            )
            normalized_x = self.normalize(self.x_data_tf)
            # 同一个 NN_L 在高保真采样点的预测；不冻结、不截断梯度。
            self.low_at_high = self.neural_net(
                normalized_x, self.low_weights, self.low_biases
            )
            features = tf.concat([normalized_x, self.low_at_high / self.ps_scale], axis=1)
            self.linear_pred = self.ps_scale * self.neural_net(
                features, self.linear_weights, self.linear_biases, nonlinear=False
            )
            self.nonlinear_pred = self.ps_scale * self.neural_net(
                features, self.nonlinear_weights, self.nonlinear_biases
            )
            self.y_pred = tf.add(
                self.alpha * self.linear_pred,
                (1.0 - self.alpha) * self.nonlinear_pred,
                name="high_fidelity_prediction",
            )
            # 每个保真度分别求均方误差，避免样本多的低保真域自动取得更大权重。
            self.loss_low = tf.reduce_mean(tf.square(self.y_low_tf - self.low_pred))
            self.loss_high = tf.reduce_mean(tf.square(self.y_data_tf - self.y_pred))
            # 仅使用低、高保真数据误差，不加入 L1 或 L2 正则项。
            self.loss = (
                LAMBDA_LOW * self.loss_low + LAMBDA_HIGH * self.loss_high
            )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, var_list=self.trainable_variables)

            # 两类数据损失都必须能够回传到低保真网络。
            low_gradients = tf.gradients(self.loss_low, self.low_variables)
            high_gradients = tf.gradients(self.loss_high, self.low_variables)
            if any(value is None for value in low_gradients + high_gradients):
                raise RuntimeError("低保真网络的联合损失梯度连接缺失")
            self.low_gradient_norms = [
                tf.global_norm(low_gradients), tf.global_norm(high_gradients)
            ]
            self.saver = tf.train.Saver(var_list=self.variables)
            initializer = tf.global_variables_initializer()

        self.feed_dict = {
            self.x_low_tf: self.x_low, self.y_low_tf: self.y_low,
            self.x_data_tf: self.x_data, self.y_data_tf: self.y_data,
        }
        self.history: list[list[float]] = []
        self.sess = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
        )
        try:
            self.sess.run(initializer)
            self.parameters_before = self.sess.run(self.trainable_variables)
            self.initial_low_gradient_norms = [
                float(value) for value in self.sess.run(
                    self.low_gradient_norms, self.feed_dict
                )
            ]
        except Exception:
            self.sess.close()
            raise
        # 不包含 alpha，供扫掠检查各组网络初值完全一致。
        network_values = self.sess.run(
            self.low_variables + self.linear_weights + self.linear_biases
            + self.nonlinear_weights + self.nonlinear_biases
        )
        self.initial_network_sha256 = hashlib.sha256(
            b"".join(value.tobytes() for value in network_values)
        ).hexdigest()

    def normalize(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def initialize_nn(self, layers):
        weights, biases = [], []
        for index, (input_dim, output_dim) in enumerate(zip(layers[:-1], layers[1:])):
            # Glorot/Xavier 正态初始化；所有网络独立初始化，不读取旧检查点。
            weights.append(tf.Variable(
                tf.random_normal(
                    [input_dim, output_dim],
                    stddev=np.sqrt(2.0 / (input_dim + output_dim)),
                ),
                name=f"weight_{index}",
            ))
            biases.append(tf.Variable(
                tf.zeros([1, output_dim], dtype=tf.float32), name=f"bias_{index}"
            ))
        return weights, biases

    def neural_net(self, x, weights, biases, nonlinear=True):
        hidden = x
        for index in range(len(weights)):
            hidden = tf.add(tf.matmul(hidden, weights[index]), biases[index])
            if nonlinear and index < len(weights) - 1:
                hidden = tf.tanh(hidden)
        return hidden

    def record(self, epoch):
        values = self.sess.run(
            [self.loss, self.loss_low, self.loss_high, self.alpha],
            self.feed_dict,
        )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"第 {epoch} 轮损失或 alpha 出现 NaN 或 Inf")
        row = [
            epoch, float(values[0]), float(LAMBDA_LOW * values[1]),
            float(LAMBDA_HIGH * values[2]), float(values[3]),
        ]
        self.history.append(row)
        if epoch in (0, 1) or epoch % PRINT_EVERY == 0:
            print(
                "Adam epoch %5d: total=%.6e, low=%.6e, high=%.6e, alpha=%.6f"
                % tuple(row)
            )

    def low_parameter_max_change(self):
        after = self.sess.run(self.low_variables)
        return max(
            float(np.max(np.abs(before - current)))
            for before, current in zip(self.parameters_before, after)
        )

    def train(self):
        self.history = []
        self.record(0)
        for epoch in range(1, MAX_EPOCHS + 1):
            self.sess.run(self.train_op, self.feed_dict)
            self.record(epoch)
        if MAX_EPOCHS > 0 and self.low_parameter_max_change() == 0.0:
            raise RuntimeError("低保真网络参数未更新，联合训练检查失败")
        if self.fixed_alpha is not None and (
            float(self.sess.run(self.alpha)) != float(np.float32(self.fixed_alpha))
        ):
            raise RuntimeError("固定 alpha 在训练中发生变化")
        return np.asarray(self.history, dtype=float)

    def predict(self, x):
        return self.sess.run(self.y_pred, {self.x_data_tf: x})

    def predict_low(self, x):
        return self.sess.run(self.low_pred, {self.x_low_tf: x})

    def predict_components(self, x):
        return self.sess.run(
            [self.low_at_high, self.linear_pred, self.nonlinear_pred, self.y_pred],
            {self.x_data_tf: x},
        )

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
        arrays[split]=frame_to_model_data(frames[split])
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
    x,_=frame_to_model_data(mapped)
    frames['lhs_target']=lhs
    arrays['lhs_target']=(x,lhs.Ps_mps.to_numpy(dtype=np.float64).reshape(-1,1))
    files.append(args.lhs_file)
    return frames,arrays,audit,files
def load_low_data(path):
    """Exact filtering/split order from MSFS747-MODEL/train_pcnn_5D.py."""
    data=pd.read_excel(path,engine='openpyxl').round(2)
    domains={'h':(457,3048,6096,9144,12192),
             'mach':(.25,.3,.4,.5,.6,.7,.8,.9),
             'm':tuple(range(240000,300001,10000)),
             'tem':tuple(range(-20,21,10)), 'dt':(70,80,90)}
    selected=data['success'].astype(str).str.upper().eq('TRUE')
    for column,values in domains.items():
        selected &= data[column].isin(values)
    filtered=data.loc[selected].copy()
    valid=np.isfinite(filtered[['mach','h','m','dt','tem','hdot']].to_numpy(dtype=np.float32)).all(axis=1)
    filtered=filtered.loc[valid].copy()
    filtered['source_excel_row']=filtered.index.to_numpy()+2
    filtered['source_file']=path.name
    filtered['Ps']=-filtered['hdot']
    filtered=filtered.rename(columns={'m':'mass','dt':'N1','tem':'dtem'}).reset_index(drop=True)
    require(len(filtered)>10,'Insufficient valid low-fidelity data')
    indices=np.arange(len(filtered))
    train,remaining=train_test_split(indices,test_size=.30,random_state=SPLIT_SEED,shuffle=True)
    validation,test=train_test_split(remaining,test_size=1./3.,random_state=SPLIT_SEED,shuffle=True)
    frames={}; arrays={}; split_labels=np.empty(len(filtered),dtype=object)
    for split,ids in [('train',train),('validation',validation),('test',test)]:
        frames[split]=filtered.iloc[ids].reset_index(drop=True)
        arrays[split]=frame_to_model_data(frames[split])
        split_labels[ids]=split
    audit=pd.DataFrame({'sample_id':indices,'excel_row':filtered.source_excel_row,'split':split_labels})
    lb=arrays['train'][0].min(axis=0); ub=arrays['train'][0].max(axis=0)
    require(np.all(ub>lb),'Invalid low-fidelity normalization bounds')
    return frames,arrays,audit,lb,ub


def fresh_model(seed,rate,arrays,low_arrays,lb,ub):
    random.seed(seed)
    np.random.seed(seed)
    return MFDNN(*low_arrays['train'],*arrays['train'],lb,ub,seed=seed,learning_rate=rate)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root',type=Path,default=PROJECT/'inputfile'/'flightdata'/'flightdata_split')
    p.add_argument('--lhs-file',type=Path,default=PROJECT/'inputfile'/'lhs_target'/'LHS_target_trim_Ps_20261028_203140_904.xlsx')
    p.add_argument('--low-file',type=Path,default=SOURCE_ROOT/'flightdata'/'out_747-800_3.xlsx')
    p.add_argument('--output',type=Path,default=ROOT/'results_10seeds_5lr')
    p.add_argument('--epochs',type=int,default=3000)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    require(args.epochs>0,'epochs must be positive')
    frames,arrays,audit,paths=load_data(args)
    low_frames,low_arrays,low_audit,lb,ub=load_low_data(args.low_file)
    paths.append(args.low_file)
    manifest=dict(model='MFDNN',initialization_seeds=INITIALIZATION_SEEDS,learning_rates=RATES,
                  epochs=args.epochs,optimizer='Adam',layers=dict(low=SOURCE_LAYERS,linear=LINEAR_LAYERS,nonlinear=NONLINEAR_LAYERS),
                  batch_size_high=len(frames['train']),batch_size_low=len(low_frames['train']),
                  regularization='none',early_stopping=False,checkpoint_selection='final_epoch',
                  alpha='trainable sigmoid(logit); initial alpha=0.5',ps_scale=PS_SCALE,
                  lambda_low=LAMBDA_LOW,lambda_high=LAMBDA_HIGH,loss='MSE_low + MSE_high',
                  flight_tracks=TRACKS,flight_round_decimals=2,lhs_rounding='none',low_round_decimals=2,
                  counts={s:len(f) for s,f in frames.items()},low_counts={s:len(f) for s,f in low_frames.items()},
                  low_split_seed=SPLIT_SEED,low_split_method='same filters and sklearn two-stage split as MSFS747-MODEL/train_pcnn_5D.py',
                  lhs_excluded=int((~audit.included_in_metrics).sum()),
                  inputs={str(x.resolve()):sha(x) for x in paths},
                  initialization='Xavier normal; zero biases; all three networks jointly trained from scratch',
                  normalization=dict(origin='low-fidelity training set only',lb=lb.tolist(),ub=ub.tolist()),
                  original_algorithm=str(ALGORITHM_ORIGIN),embedded_algorithm_sha256=EMBEDDED_ALGORITHM_SHA256,
                  script_sha256=sha(__file__),python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,pandas=pd.__version__)
    print('High fidelity:',manifest['counts'],'Low fidelity:',manifest['low_counts'],flush=True)
    print('Initialization seeds:',INITIALIZATION_SEEDS,'Learning rates:',RATES,'Total runs: 50',flush=True)
    if args.check_only:
        hashes={}
        for seed in INITIALIZATION_SEEDS:
            same=[]
            for rate in RATES:
                model=fresh_model(seed,rate,arrays,low_arrays,lb,ub)
                try:
                    same.append(model.initial_network_sha256)
                    for split,(x,y) in arrays.items(): metric(y,model.predict(x))
                    for split,(x,y) in low_arrays.items(): metric(y,model.predict_low(x))
                    require(all(np.isfinite(model.initial_low_gradient_norms)) and min(model.initial_low_gradient_norms)>0,'Joint gradients invalid')
                finally: model.sess.close()
            require(len(set(same))==1,f'Initial weights differ across rates: {seed}')
            hashes[seed]=same[0]
            print(f'CHECK OK: seed {seed}; identical initial weights across all 5 rates; joint gradients connected',flush=True)
        require(len(set(hashes.values()))==10,'Initial weights do not differ across seeds')
        print('CHECK PASSED: 50 combinations; no optimizer updates; no result files written.',flush=True)
        return
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        require(args.resume,'Output exists: choose another --output or use --resume')
        old=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
        require(same_settings(old, manifest),'Resume refused: settings, inputs, or code changed')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'experiment_config.json',manifest)
    for split,f in frames.items(): f.to_csv(out/f'{split}_data_used.csv',index=False,encoding='utf-8-sig')
    for split,f in low_frames.items(): f.to_csv(out/f'low_{split}_data_used.csv',index=False,encoding='utf-8-sig')
    audit.to_csv(out/'lhs_input_audit.csv',index=False,encoding='utf-8-sig')
    low_audit.to_csv(out/'low_data_split.csv',index=False,encoding='utf-8-sig')
    all_rows=[]; all_low_rows=[]; seed_hashes={}
    for seed in INITIALIZATION_SEEDS:
        for rate in RATES:
            run=out/f'seed_{seed}'/f'lr_{rate:g}'
            marker=run/'completed.json'
            if marker.exists():
                saved=json.loads(marker.read_text(encoding='utf-8'))
                require(all((run/f).is_file() and sha(run/f)==h for f,h in saved['files'].items()),f'Completed files changed: {run}')
                all_rows.extend(pd.read_csv(run/'metrics.csv').to_dict('records'))
                all_low_rows.extend(pd.read_csv(run/'low_metrics.csv').to_dict('records'))
                pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(all_low_rows).to_csv(out/'all_low_run_metrics.csv',index=False,encoding='utf-8-sig')
                print(f'SKIP completed: seed {seed}, lr={rate:g}',flush=True)
                continue
            run.mkdir(parents=True,exist_ok=True)
            model=fresh_model(seed,rate,arrays,low_arrays,lb,ub)
            try:
                initial_hash=model.initial_network_sha256
                require(seed_hashes.setdefault(seed,initial_hash)==initial_hash,'Initialization differs across learning rates')
                write_json(run/'run_config.json',dict(model='MFDNN',initialization_seed=seed,learning_rate=rate,epochs=args.epochs,
                           initial_network_sha256=initial_hash,lb=lb.tolist(),ub=ub.tolist(),
                           alpha_initial=float(model.sess.run(model.alpha)),loss='MSE_low + MSE_high'))
                np.savez(run/'normalization_params.npz',lb=lb,ub=ub,
                         input_order=np.asarray(['mach','height_scaled','mass_scaled','N1_scaled','temperature_scaled']),
                         height_scale=HEIGHT_SCALE,mass_scale=MASS_SCALE,dt_scale=DT_SCALE,
                         celsius_to_kelvin=CELSIUS_TO_KELVIN,standard_temperature_k=STANDARD_TEMPERATURE_K,ps_scale=PS_SCALE)
                before={s:model.predict(x).ravel() for s,(x,y) in arrays.items()}
                low_before={s:model.predict_low(x).ravel() for s,(x,y) in low_arrays.items()}
                history=[]; start=time.perf_counter()
                with (run/'run.log').open('w',encoding='utf-8',buffering=1) as log:
                    for epoch in range(args.epochs+1):
                        if epoch: model.sess.run(model.train_op,model.feed_dict)
                        total,low_loss,high_loss,alpha=model.sess.run([model.loss,model.loss_low,model.loss_high,model.alpha],model.feed_dict)
                        val=metric(arrays['validation'][1],model.predict(arrays['validation'][0]))['mse']
                        require(np.isfinite([total,low_loss,high_loss,alpha,val]).all(),'Nonfinite training loss')
                        history.append(dict(epoch=epoch,total_loss=float(total),low_train_mse=float(low_loss),
                                            train_mse=float(high_loss),validation_mse=val,alpha=float(alpha)))
                        if epoch%500==0 or epoch==args.epochs:
                            msg=f'seed={seed} lr={rate:g} epoch={epoch} total={total:.8g} low={low_loss:.8g} high={high_loss:.8g} val={val:.8g} alpha={alpha:.6f}'
                            print(msg,flush=True); log.write(msg+'\n')
                            pd.DataFrame(history).to_csv(run/'history.csv',index=False)
                elapsed=time.perf_counter()-start
                require(model.low_parameter_max_change()>0,'Low-fidelity network did not update')
                rows=[]; low_rows=[]
                for prefix,fs,ds,initial,destination in [('',frames,arrays,before,rows),('low_',low_frames,low_arrays,low_before,low_rows)]:
                    for split,(x,y) in ds.items():
                        pred=(model.predict_low(x) if prefix else model.predict(x)).ravel()
                        row=dict(initialization_seed=seed,learning_rate=rate,split=split,**metric(y,pred),
                                 elapsed_seconds=elapsed,alpha_final=float(alpha),initial_network_sha256=initial_hash)
                        row.update({f'initial_{k}':v for k,v in metric(y,initial[split]).items() if k!='n'})
                        destination.append(row)
                        f=fs[split].copy(); f['ps_true']=y.ravel(); f['ps_pred_before']=initial[split]; f['ps_pred']=pred
                        f['error_pred_minus_true']=pred.astype(np.float64)-y.ravel().astype(np.float64)
                        f['absolute_error']=abs(f.error_pred_minus_true)
                        if not prefix:
                            low,linear,nonlinear,_=model.predict_components(x)
                            f['low_at_high']=low.ravel(); f['linear_prediction']=linear.ravel(); f['nonlinear_prediction']=nonlinear.ravel()
                        f.to_csv(run/f'{prefix}{split}_predictions.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(low_rows).to_csv(run/'low_metrics.csv',index=False,encoding='utf-8-sig')
                model.saver.save(model.sess,str(run/'model.ckpt'),write_meta_graph=False)
                write_json(marker,dict(status='complete',files={f.name:sha(f) for f in run.iterdir() if f.is_file() and f.name!='completed.json'}))
                all_rows.extend(rows); all_low_rows.extend(low_rows)
                pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(all_low_rows).to_csv(out/'all_low_run_metrics.csv',index=False,encoding='utf-8-sig')
            except Exception as exc:
                (run/'error.txt').write_text(repr(exc),encoding='utf-8'); raise
            finally: model.sess.close()
    require(len(all_rows)==200 and len(all_low_rows)==150,'Incomplete metric rows')
    print(f'FINISHED: 50 models; 200 high-fidelity + 150 low-fidelity metric rows. Results: {out}',flush=True)

if __name__=='__main__':
    main()

