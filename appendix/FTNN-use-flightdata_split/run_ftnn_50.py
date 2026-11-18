"""10个源模型 × 5个学习率，独立进行50次FTNN微调；不自动选型或平均。

运行：python run_ftnn_50.py
仅检查输入、权重及预测接口（不训练、不写结果）：python run_ftnn_50.py --check-only
中断后跳过已完整完成的运行：python run_ftnn_50.py --resume

默认3000轮，全批量Adam，无正则项/早停，固定微调种子1234。
每次均加载对应源模型原始权重，重新初始化Adam；不沿用前一学习率结果。
输出 results_10sources_5lr/source_1000/lr_0.0001/ 等50个目录。
每次保存：history.csv、metrics.csv、四类逐点预测、model.ckpt、
normalization_params.npz、run_config.json、run.log、completed.json。
根目录 all_run_metrics.csv 为逐运行明细，不是跨种子平均。
独立脚本，已内置原FTNN网络与训练损失，不依赖同目录其他文件。
环境依赖：tensorflow 2.15.1、numpy、pandas、openpyxl。
"""

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
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 原FTNN算法：5-64-64-64-1、tanh隐藏层、线性输出、全批量Adam、预测MSE。
# 保持变量创建顺序与原检查点一致：Variable=W0, Variable_1=b0, ...。
class DataFittingNN:
    def __init__(self,x_data,y_data,lb,ub):
        self.lb,self.ub=lb,ub
        self.weights,self.biases=[],[]
        for left,right in zip(alg.LAYERS[:-1],alg.LAYERS[1:]):
            self.weights.append(tf.Variable(tf.truncated_normal(
                [left,right],stddev=np.sqrt(2.0/(left+right)))))
            self.biases.append(tf.Variable(tf.zeros([1,right],dtype=tf.float32)))
        self.variables=list(self.weights)+list(self.biases)
        self.sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
        self.x_data_tf=tf.placeholder(tf.float32,shape=[None,5])
        self.y_data_tf=tf.placeholder(tf.float32,shape=[None,1])
        hidden=2.0*(self.x_data_tf-self.lb)/(self.ub-self.lb)-1.0
        for weight,bias in zip(self.weights[:-1],self.biases[:-1]):
            hidden=tf.tanh(tf.add(tf.matmul(hidden,weight),bias))
        self.y_pred=tf.add(tf.matmul(hidden,self.weights[-1]),self.biases[-1])
        self.loss_data=tf.reduce_mean(tf.square(self.y_data_tf-self.y_pred))
        self.loss=alg.LAMBDA_DATA*self.loss_data
        self.optimizer=tf.train.AdamOptimizer(learning_rate=alg.LEARNING_RATE)
        self.train_op=self.optimizer.minimize(self.loss,var_list=self.variables)
        self.feed_dict={self.x_data_tf:x_data,self.y_data_tf:y_data}
        self.sess.run(tf.global_variables_initializer())

    def predict(self,x):
        return self.sess.run(self.y_pred,{self.x_data_tf:x})

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

alg=SimpleNamespace(tf=tf,LAYERS=[5,64,64,64,1],LEARNING_RATE=1e-3,LAMBDA_DATA=1.,
                    DataFittingNN=DataFittingNN,frame_to_model_data=frame_to_model_data)

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'data'
SOURCE_ROOT = PROJECT/'MSFS747-MODEL'/'out_pcnn_5D'
RATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SOURCE_SEEDS = list(range(1000,1010))
TRACKS = ['0','2.5','6','12','25']
FT_SEED = 1234

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

def fresh_model(source,rate,arrays):
    alg.tf.reset_default_graph()
    random.seed(FT_SEED)
    np.random.seed(FT_SEED)
    alg.tf.set_random_seed(FT_SEED)
    alg.LEARNING_RATE=rate
    alg.LAMBDA_DATA=1.
    model=alg.DataFittingNN(*arrays['train'],source['lb'],source['ub'])
    # Network variables only: Adam moments remain freshly initialized at zero.
    model.sess.run([v.assign(source['weights'][v.op.name]) for v in model.variables])
    for v,actual in zip(model.variables,model.sess.run(model.variables)):
        require(np.array_equal(actual,source['weights'][v.op.name]),'Source weight loading failed')
    return model

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--data-root',type=Path,default=PROJECT/'inputfile'/'flightdata'/'flightdata_split')
    p.add_argument('--lhs-file',type=Path,default=PROJECT/'inputfile'/'lhs_target'/'LHS_target_trim_Ps_20261028_203140_904.xlsx')
    p.add_argument('--output',type=Path,default=ROOT/'results_10sources_5lr')
    p.add_argument('--epochs',type=int,default=3000)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    require(args.epochs>0,'epochs must be positive')
    frames,arrays,audit,paths=load_data(args)
    sources=load_sources(args)
    manifest=dict(source_seeds=SOURCE_SEEDS,learning_rates=RATES,ft_seed=FT_SEED,
                  epochs=args.epochs,optimizer='Adam',layers=alg.LAYERS,batch_size=len(frames['train']),
                  regularization='none',early_stopping=False,checkpoint_selection='final_epoch',
                  flight_tracks=TRACKS,flight_round_decimals=2,lhs_rounding='none',
                  counts={s:len(f) for s,f in frames.items()},lhs_excluded=int((~audit.included_in_metrics).sum()),
                  inputs={str(x.resolve()):sha(x) for x in paths},
                  sources=[dict(seed=s['seed'],weights_sha256=s['weight_hash'],files=s['files']) for s in sources],
                  script_sha256=sha(__file__),algorithm='embedded original FTNN architecture, MSE, Adam',
                  python=platform.python_version(),tensorflow=alg.tf.__version__,numpy=np.__version__,pandas=pd.__version__)
    print('Data:',manifest['counts'],flush=True)
    print('Source seeds:',SOURCE_SEEDS,'; learning rates:',RATES,'; total runs: 50',flush=True)
    if args.check_only:
        for source in sources:
            model=fresh_model(source,RATES[0],arrays)
            try:
                for split,(x,y) in arrays.items():
                    metric(y,model.predict(x))
                print(f'CHECK OK: source {source["seed"]}, normalization and prediction interface',flush=True)
            finally:
                model.sess.close()
        print('CHECK PASSED. No optimizer updates executed; no result directory written.',flush=True)
        return
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        require(args.resume,'Output exists: choose a new --output directory or use --resume')
        old=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
        require(same_settings(old, manifest),'Resume refused: experiment settings, source files, or code changed')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'experiment_config.json',manifest)
    for split,f in frames.items():
        f.to_csv(out/f'{split}_data_used.csv',index=False,encoding='utf-8-sig')
    audit.to_csv(out/'lhs_input_audit.csv',index=False,encoding='utf-8-sig')
    all_rows=[]
    for source in sources:
        for rate in RATES:
            run=out/f'source_{source["seed"]}'/f'lr_{rate:g}'
            marker=run/'completed.json'
            if marker.exists():
                saved=json.loads(marker.read_text(encoding='utf-8'))
                require(all((run/f).is_file() and sha(run/f)==h for f,h in saved['files'].items()),f'Completed files changed: {run}')
                all_rows.extend(pd.read_csv(run/'metrics.csv').to_dict('records'))
                pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
                print(f'SKIP completed: {run.name}, source {source["seed"]}',flush=True)
                continue
            run.mkdir(parents=True,exist_ok=True)
            model=fresh_model(source,rate,arrays)
            try:
                write_json(run/'run_config.json',dict(source_seed=source['seed'],learning_rate=rate,
                           ft_seed=FT_SEED,epochs=args.epochs,source_directory=str(source['directory'].resolve()),
                           source_weights_sha256=source['weight_hash'],source_config=source['config'],
                           lb=source['lb'].tolist(),ub=source['ub'].tolist(),loss='prediction MSE only'))
                shutil.copy2(source['directory']/'normalization_params.npz',run/'normalization_params.npz')
                before={s:model.predict(x).ravel() for s,(x,y) in arrays.items()}
                initial_hash=hashlib.sha256(before['train'].tobytes()).hexdigest()
                history=[]
                start=time.perf_counter()
                with (run/'run.log').open('w',encoding='utf-8',buffering=1) as log:
                    log.write(f'source={source["seed"]}, lr={rate:g}, epochs={args.epochs}\n')
                    for epoch in range(args.epochs+1):
                        if epoch:
                            model.sess.run(model.train_op,feed_dict=model.feed_dict)
                        train_loss=float(model.sess.run(model.loss_data,feed_dict=model.feed_dict))
                        xval,yval=arrays['validation']
                        val_loss=metric(yval,model.predict(xval))['mse']
                        require(np.isfinite(train_loss) and np.isfinite(val_loss),'Nonfinite training loss')
                        history.append(dict(epoch=epoch,train_mse=train_loss,validation_mse=val_loss))
                        if epoch%500==0 or epoch==args.epochs:
                            msg=f'source={source["seed"]} lr={rate:g} epoch={epoch} train={train_loss:.8g} val={val_loss:.8g}'
                            print(msg,flush=True)
                            log.write(msg+'\n')
                            pd.DataFrame(history).to_csv(run/'history.csv',index=False)
                elapsed=time.perf_counter()-start
                rows=[]
                for split,(x,y) in arrays.items():
                    pred=model.predict(x).ravel()
                    row=dict(source_seed=source['seed'],learning_rate=rate,ft_seed=FT_SEED,split=split,
                             **metric(y,pred),elapsed_seconds=elapsed,initial_prediction_sha256=initial_hash)
                    row.update({f'initial_{k}':v for k,v in metric(y,before[split]).items() if k!='n'})
                    rows.append(row)
                    f=frames[split].copy()
                    f['ps_true']=y.ravel()
                    f['ps_pred_before']=before[split]
                    f['ps_pred']=pred
                    f['error_pred_minus_true']=pred.astype(np.float64)-y.ravel().astype(np.float64)
                    f['absolute_error']=abs(f.error_pred_minus_true)
                    f.to_csv(run/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False,encoding='utf-8-sig')
                alg.tf.train.Saver(var_list=model.variables).save(model.sess,str(run/'model.ckpt'),write_meta_graph=False)
                write_json(marker,dict(status='complete',files={f.name:sha(f) for f in run.iterdir() if f.is_file() and f.name!='completed.json'}))
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(out/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
            except Exception as exc:
                (run/'error.txt').write_text(repr(exc),encoding='utf-8')
                raise
            finally:
                model.sess.close()
    require(len(all_rows)==200,'Expected 50 runs times 4 evaluation datasets')
    print(f'FINISHED: 50 models, 200 metric rows. Results: {out}',flush=True)

if __name__=='__main__':
    main()
