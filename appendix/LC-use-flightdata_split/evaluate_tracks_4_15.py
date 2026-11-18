"""独立评估未参与训练的4°、15°完整航迹，不重训、不拟合、不选择学习率。
运行：python evaluate_tracks_4_15.py
只校验并推理，不写文件：python evaluate_tracks_4_15.py --check-only
两条航迹分别合并train/validation/test共200点，按t排序，数值四舍五入至2位。
每次从已保存检查点和归一化参数构建纯推理图，没有优化器。
先复现原验证集预测以核对模型加载，再评估未见航迹。
输出独立目录heldout_tracks_4_15：1200 dpi PNG、逐点预测、每次指标、均值和样本标准差。
RMSE、MAE先按各次运行分别计算，再对10次取均值；ddof=1，不是置信区间。
"""

from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
os.environ['TF_NUM_INTEROP_THREADS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import hashlib
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
tf.disable_v2_behavior()

MODEL='LC'
RESULTS_DIRECTORY='results_10sources_lc'
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent / 'data'
SEEDS=list(range(1000,1010))
RATES=[1e-4,5e-4,1e-3,5e-3,1e-2]
TRACKS=['4','15']


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(frame):
    m=frame.mach.to_numpy(dtype=np.float32)
    h=frame.h.to_numpy(dtype=np.float32)
    mass=frame.mass.to_numpy(dtype=np.float32)
    n1=frame.N1.to_numpy(dtype=np.float32)
    temp=frame.dtem.to_numpy(dtype=np.float32)
    return np.column_stack([m,h/10000.,mass/100000.,n1/100.,(temp+273.15)/273.15]).astype(np.float32)


def load_tracks(root):
    frames={}; paths=[]
    for track in TRACKS:
        parts=[]
        for split,n in [('train',140),('validation',40),('test',20)]:
            found=list(root.glob(f'out_20_{track}_90_70_*_{split}.xlsx'))
            require(len(found)==1,f'Expected one file: track {track}, split {split}: {found}')
            path=found[0]; f=pd.read_excel(path,engine='openpyxl').round(2)
            require(len(f)==n,f'Unexpected count: {path}')
            require(np.isfinite(f[['t','mach','h','mass','N1','dtem','V','Ps']].to_numpy()).all(),f'Invalid data: {path}')
            require((f.V>0).all(),f'Invalid velocity: {path}')
            f['track']=track; f['original_split']=split; f['source_file']=path.name
            f['source_excel_row']=np.arange(len(f))+2
            parts.append(f); paths.append(path)
        frame=pd.concat(parts,ignore_index=True).sort_values('t').reset_index(drop=True)
        require(len(frame)==200 and not frame.t.duplicated().any(),f'Duplicate/missing times: track {track}')
        frames[track]=frame
    return frames,paths


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64).ravel(); p=np.asarray(p,dtype=np.float64).ravel()
    require(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid predictions')
    error=p-y; sst=np.sum((y-y.mean())**2); require(sst>0,'Constant target')
    return dict(n=len(y),mse=float(np.mean(error**2)),rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(abs(error))),r2=float(1.-np.sum(error**2)/sst))


class Predictor:
    """Checkpoint constants only; no trainable variables, random ops or optimizer."""
    def __init__(self,run):
        with np.load(run/'normalization_params.npz',allow_pickle=False) as z:
            lb=z['lb'].astype(np.float32); ub=z['ub'].astype(np.float32)
            require(lb.shape==(5,) and ub.shape==(5,) and np.isfinite([lb,ub]).all() and (ub>lb).all(),'Invalid normalization')
            require(z['input_order'].tolist()==['mach','height_scaled','mass_scaled','N1_scaled','temperature_scaled'],'Input order mismatch')
            for key,value in dict(height_scale=10000.,mass_scale=100000.,dt_scale=100.,celsius_to_kelvin=273.15,standard_temperature_k=273.15).items():
                require(np.isclose(float(z[key]),value),f'Unexpected scaling: {key}')
        with tempfile.TemporaryDirectory(prefix='heldout_ckpt_') as tmp:
            for path in run.glob('model.ckpt.*'):
                if path.suffix=='.index' or '.data-' in path.name: shutil.copy2(path,Path(tmp)/path.name)
            reader=tf.train.load_checkpoint(str(Path(tmp)/'model.ckpt'))
            tensors={name:reader.get_tensor(name) for name in reader.get_variable_to_shape_map()}
            del reader
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.x=tf.placeholder(tf.float32,[None,5])
            normalized=2.0*(self.x-lb)/(ub-lb)-1.0
            def network(x,scope=None,layers=4,nonlinear=True):
                hidden=x
                for i in range(layers):
                    wn=f'{scope}/weight_{i}' if scope else ('Variable' if i==0 else f'Variable_{2*i}')
                    bn=f'{scope}/bias_{i}' if scope else f'Variable_{2*i+1}'
                    w=tensors[wn]; b=tensors[bn]
                    require(w.dtype==np.float32 and b.dtype==np.float32 and np.isfinite(w).all() and np.isfinite(b).all(),'Invalid model tensor')
                    hidden=tf.add(tf.matmul(hidden,tf.constant(w)),tf.constant(b))
                    if nonlinear and i<layers-1: hidden=tf.tanh(hidden)
                return hidden
            if MODEL in ['FTNN','DNN']:
                pred=network(normalized)
            elif MODEL=='RCNN':
                pred=tf.add(network(normalized,'source_model'),network(normalized,'residual_model'))
            elif MODEL=='LC':
                source=network(normalized,'source_model')
                pred=tf.add(tf.constant(tensors['calibration/a'])*tf.cast(source,tf.float64),tf.constant(tensors['calibration/b']))
            elif MODEL=='MFDNN':
                low=network(normalized,'low_fidelity')
                scale=tf.constant(tensors['multifidelity/ps_scale'])
                features=tf.concat([normalized,low/scale],axis=1)
                linear=scale*network(features,'high_fidelity_linear',layers=2,nonlinear=False)
                nonlinear=scale*network(features,'high_fidelity_nonlinear')
                alpha=(tf.sigmoid(tf.constant(tensors['multifidelity/alpha_logit']))
                       if 'multifidelity/alpha_logit' in tensors else tf.constant(tensors['multifidelity/alpha_fixed']))
                pred=tf.add(alpha*linear,(1.0-alpha)*nonlinear)
            else: raise ValueError(MODEL)
            self.pred=pred
        self.session=tf.Session(graph=self.graph)

    def predict(self,x):
        return self.session.run(self.pred,{self.x:x}).ravel()

    def close(self):
        self.session.close()


def summarize(frame):
    records=[]
    for (track,rate),group in frame.groupby(['track','learning_rate'],sort=False,dropna=False):
        require(len(group)==10 and set(group.seed)==set(SEEDS),'Incomplete seed group')
        row=dict(model=MODEL,track=track,learning_rate=rate,seed_count=10,sample_count=200)
        for metric in ['mse','rmse','mae','r2']:
            v=group[metric].to_numpy(dtype=float)
            row.update({metric+'_mean':float(v.mean()),metric+'_std':float(v.std(ddof=1)),
                        metric+'_min':float(v.min()),metric+'_max':float(v.max())})
        records.append(row)
    return pd.DataFrame(records)


def draw(summary,frame,path):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,track in zip(axes,TRACKS):
        group=summary[summary.track==track].sort_values('learning_rate')
        for i,(metric,color,fmt) in enumerate([('rmse','#166b96','o-'),('mae','#cb7024','s--')]):
            if MODEL=='LC':
                row=group.iloc[0]
                ax.bar(i,row[metric+'_mean'],width=.5,color=color,alpha=.25)
                ax.errorbar(i,row[metric+'_mean'],yerr=row[metric+'_std'],fmt='D',color=color,capsize=4,zorder=4)
                values=frame[frame.track==track].sort_values('seed')[metric]
                ax.scatter(i+np.linspace(-.16,.16,10),values,color=color,s=20,zorder=3)
            else:
                ax.errorbar(group.learning_rate,group[metric+'_mean'],yerr=group[metric+'_std'],
                            fmt=fmt,color=color,markersize=5,label=metric.upper(),linewidth=1.8,capsize=4)
        if MODEL=='LC':
            ax.set_xticks([0,1],['RMSE','MAE']); ax.set_xlim(-.55,1.55)
        else:
            ax.set_xscale('log'); ax.xaxis.set_major_locator(FixedLocator(RATES))
            ax.xaxis.set_major_formatter(FixedFormatter(['1e-4','5e-4','1e-3','5e-3','1e-2']))
            ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlabel('Learning rate (log scale)')
            ax.legend(frameon=False)
        ax.set_title(f'Held-out trajectory {track}° (n=200)')
        ax.set_ylabel('Prediction error (m/s)'); ax.grid(True,alpha=.22,linewidth=.7)
        ax.spines[['top','right']].set_visible(False); ax.margins(x=.1,y=.15)
    fig.suptitle(f'{MODEL}: unseen trajectories 4° and 15°',fontsize=14,y=.98)
    fig.text(.5,.025,'Mean of 10 runs; error bars: ±1 sample SD (ddof = 1), not confidence intervals.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93],w_pad=2.2)
    fig.savefig(path,dpi=1200,facecolor='white'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results',type=Path,default=ROOT/RESULTS_DIRECTORY)
    parser.add_argument('--data-root',type=Path,default=PROJECT/'inputfile'/'flightdata'/'flightdata_split')
    parser.add_argument('--output',type=Path,default=ROOT/'heldout_tracks_4_15')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    config_path=args.results/'experiment_config.json'; config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    require(config.get('model','FTNN')==MODEL,'Wrong model experiment')
    require(set(map(str,config['flight_tracks']))=={'0','2.5','6','12','25'},'Training tracks differ: held-out status cannot be confirmed')
    require(config.get('source_seeds',config.get('initialization_seeds'))==SEEDS,'Wrong seeds')
    if MODEL!='LC': require(config['learning_rates']==RATES,'Wrong learning rates')
    if MODEL=='RCNN': require(config.get('residual_seed_rule')=='source_seed','Expected source-matched residual initialization experiment')
    frames,paths=load_tracks(args.data_root)
    baseline_file=args.results/'validation_data_used.csv'
    baseline=pd.read_csv(baseline_file); xb=inputs(baseline)
    records=[]; predictions=[]; audits=[]
    rates=[None] if MODEL=='LC' else RATES
    for seed in SEEDS:
        for rate in rates:
            run=args.results/f'{"seed" if MODEL in ["DNN","MFDNN"] else "source"}_{seed}'
            if rate is not None: run=run/f'lr_{rate:g}'
            marker=run/'completed.json'; require(marker.is_file(),f'Unfinished run: {run}')
            saved=json.loads(marker.read_text(encoding='utf-8'))
            names=['model.ckpt.index','normalization_params.npz','run_config.json','validation_predictions.csv']
            names += [x.name for x in sorted(run.glob('model.ckpt.data-*'))]
            require(any('.data-' in n for n in names),'Missing checkpoint data')
            for name in names:
                require(name in saved['files'] and (run/name).is_file() and sha(run/name)==saved['files'][name],f'Changed result file: {run/name}')
            cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
            require(cfg.get('source_seed',cfg.get('initialization_seed'))==seed,'Run seed mismatch')
            if rate is not None: require(cfg['learning_rate']==rate,'Run learning rate mismatch')
            predictor=Predictor(run)
            try:
                reference=pd.read_csv(run/'validation_predictions.csv')
                require(np.allclose(inputs(reference),xb,rtol=0,atol=0),'Validation sample order mismatch')
                baseline_pred=predictor.predict(xb)
                require(np.allclose(baseline_pred,reference.ps_pred,rtol=2e-5,atol=2e-5),f'Checkpoint inference differs from saved predictions: {run}')
                audits.append(dict(seed=seed,learning_rate=rate,run=str(run.resolve()),
                                   files={name:saved['files'][name] for name in names},
                                   validation_max_abs_difference=float(np.max(abs(baseline_pred-reference.ps_pred.to_numpy())))))
                for track,f in frames.items():
                    pred=predictor.predict(inputs(f))
                    # Identical target precision to the earlier flight evaluations.
                    y=f.Ps.to_numpy(dtype=np.float32).astype(np.float64)
                    records.append(dict(model=MODEL,seed=seed,learning_rate=rate,track=track,**metrics(y,pred)))
                    detail=f.copy(); detail['model']=MODEL; detail['seed']=seed; detail['learning_rate']=rate
                    detail['ps_true']=y; detail['ps_pred']=pred.astype(np.float64)
                    detail['error_pred_minus_true']=pred.astype(np.float64)-y; detail['absolute_error']=abs(detail.error_pred_minus_true)
                    predictions.append(detail)
            finally: predictor.close()
        print(f'{MODEL}: seed {seed} complete; checkpoint predictions verified',flush=True)
    data=pd.DataFrame(records); summary=summarize(data)
    require(len(data)==(20 if MODEL=='LC' else 100),'Incomplete evaluation')
    if args.check_only:
        print('CHECK PASSED: prediction-only evaluation; no files written.'); return
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    for track,f in frames.items(): f.to_csv(output/f'track_{track}_data_used.csv',index=False,encoding='utf-8-sig')
    data.to_csv(output/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(output/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    pd.concat(predictions,ignore_index=True).to_csv(output/'all_point_predictions.csv',index=False,encoding='utf-8-sig')
    draw(summary,data,output/'tracks_4_15_rmse_mae.png')
    manifest=dict(model=MODEL,source_results=str(args.results.resolve()),
                  inputs={str(p.resolve()):sha(p) for p in paths},experiment_config_sha256=sha(config_path),
                  validation_data_sha256=sha(baseline_file),source_runs=audits,script_sha256=sha(__file__),
                  tracks=TRACKS,points_per_track=200,round_decimals=2,source_splits_combined=['train','validation','test'],
                  training_performed=False,model_selection_performed=False,ddof=1,
                  statistic='mean of individual run metrics; error bars are sample SD, not confidence intervals',
                  dpi=1200,tensorflow=tf.__version__,numpy=np.__version__,pandas=pd.__version__)
    (output/'evaluation_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(summary[['track','learning_rate','rmse_mean','rmse_std','mae_mean','mae_std']].to_string(index=False))
    print(f'FINISHED: {output}',flush=True)

if __name__=='__main__':
    main()
