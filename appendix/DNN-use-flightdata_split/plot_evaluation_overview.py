"""精简三栏图：已见航迹测试点、未见4°+15°合并400点、目标域LHS。
运行 python plot_evaluation_overview.py。仅读取已保存结果，不重训。
未见航迹先逐次合并400点计算指标，再对10次指标取均值和样本标准差ddof=1。
输出evaluation_overview目录：1200dpi PNG、逐次指标、汇总CSV、作图配置。
已见指航迹参与过训练，不表示这些测试样本参与过训练。
LHS评估覆盖目标域，不能仅因是LHS就认定所有点都在训练输入范围之外。
"""

from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
ROOT=Path(__file__).resolve().parent
MODEL='DNN'
RESULTS_DIRECTORY='results_10seeds_5lr'
SEEDS=list(range(1000,1010))
RATES=[1e-4,5e-4,1e-3,5e-3,1e-2]
SPLITS=['test','unseen_tracks','lhs_target']
TITLES=['Seen trajectories: test','Unseen trajectories: 4° + 15°','Target-domain LHS']

def require(c,m):
    if not c: raise ValueError(m)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    require(y.shape==p.shape and np.isfinite([y,p]).all(),'Invalid predictions')
    e=p-y; mse=np.mean(e**2); sst=np.sum((y-y.mean())**2)
    require(sst>0,'Constant reference')
    return dict(n=len(y),mse=float(mse),rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(e))),r2=float(1-np.sum(e**2)/sst))

def select(frame,seed,rate):
    return frame[(frame.seed==seed)&(frame.learning_rate.isna() if rate is None else frame.learning_rate==rate)]

def load_data(results,heldout):
    config_path=results/'experiment_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    audit=json.loads((heldout/'evaluation_config.json').read_text(encoding='utf-8'))
    require(config.get('model','FTNN')==MODEL and audit['model']==MODEL,'Wrong model')
    require(audit['experiment_config_sha256']==sha(config_path),'Held-out predictions belong to a different experiment')
    require(config.get('source_seeds',config.get('initialization_seeds'))==SEEDS,'Wrong seeds')
    if MODEL!='LC': require(config['learning_rates']==RATES,'Wrong rates')
    original=pd.read_csv(results/'all_run_metrics.csv').rename(columns={'source_seed':'seed','initialization_seed':'seed'})
    if MODEL=='LC': original['learning_rate']=np.nan
    require(not original.duplicated(['seed','learning_rate','split']).any(),'Duplicate original metrics')
    points=pd.read_csv(heldout/'all_point_predictions.csv')
    separate=pd.read_csv(heldout/'all_run_metrics.csv')
    rates=[None] if MODEL=='LC' else RATES
    count=10*len(rates)
    require(len(points)==count*400 and len(original)==count*4 and len(separate)==count*2,'Incomplete data')
    require(set(points.model)=={MODEL} and set(points.seed)==set(SEEDS),'Wrong held-out model/seeds')
    rows=[]
    for seed in SEEDS:
        for rate in rates:
            run=results/f'{"seed" if MODEL in ["DNN","MFDNN"] else "source"}_{seed}'
            if rate is not None: run=run/f'lr_{rate:g}'
            marker=json.loads((run/'completed.json').read_text(encoding='utf-8'))
            matched=[r for r in audit['source_runs'] if r['seed']==seed and r['learning_rate']==rate]
            require(len(matched)==1,'Missing held-out model audit')
            for name,h in matched[0]['files'].items():
                require(sha(run/name)==h and marker['files'][name]==h,'Model changed since held-out evaluation')
            require(sha(run/'metrics.csv')==marker['files']['metrics.csv'],'Changed original metrics')
            saved=pd.read_csv(run/'metrics.csv')
            for split in ['test','lhs_target']:
                part=select(original,seed,rate); part=part[part.split==split]
                require(len(part)==1,'Missing original metric')
                row=part.iloc[0]; reference=saved[saved.split==split].iloc[0]
                keys=['mse','rmse','mae','r2','n']
                require(np.allclose(row[keys].to_numpy(dtype=float),reference[keys].to_numpy(dtype=float),rtol=1e-12,atol=1e-12),'Original aggregate differs from run metrics')
                require(row.n==config['counts'][split],'Unexpected sample count')
                rows.append(dict(model=MODEL,seed=seed,learning_rate=rate,split=split,**{k:float(row[k]) for k in keys}))
            group=select(points,seed,rate)
            require(len(group)==400 and group.groupby('track').size().to_dict()=={4:200,15:200},'Expected 200 points per unseen trajectory')
            require(not group.duplicated(['track','t']).any(),'Repeated held-out points')
            result=metrics(group.ps_true,group.ps_pred)
            # Independent pooled MSE/MAE check using the earlier two 200-point evaluations.
            previous=select(separate,seed,rate)
            require(len(previous)==2,'Missing separate trajectory metrics')
            require(np.allclose([result['mse'],result['mae']],np.average(previous[['mse','mae']],axis=0,weights=previous.n),rtol=1e-10,atol=1e-10),'Pooled metric check failed')
            rows.append(dict(model=MODEL,seed=seed,learning_rate=rate,split='unseen_tracks',**result))
    data=pd.DataFrame(rows); summary=[]
    for split in SPLITS:
        for rate in rates:
            group=data[(data.split==split)&(data.learning_rate.isna() if rate is None else data.learning_rate==rate)]
            require(len(group)==10 and set(group.seed)==set(SEEDS),'Incomplete seed group')
            row=dict(model=MODEL,split=split,learning_rate=rate,seed_count=10,sample_count=int(group.n.iloc[0]))
            for metric in ['mse','rmse','mae','r2']:
                v=group[metric].to_numpy(dtype=float)
                row[metric+'_mean']=float(v.mean()); row[metric+'_std']=float(v.std(ddof=1))
            summary.append(row)
    return data,pd.DataFrame(summary)

def draw(data,summary,path):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,split,title in zip(axes,SPLITS,TITLES):
        group=summary[summary.split==split].sort_values('learning_rate')
        for i,(key,color,fmt) in enumerate([('rmse','#166b96','o-'),('mae','#cb7024','s--')]):
            if MODEL=='LC':
                row=group.iloc[0]
                ax.bar(i,row[key+'_mean'],width=.5,color=color,alpha=.25)
                ax.errorbar(i,row[key+'_mean'],yerr=row[key+'_std'],fmt='D',color=color,capsize=4,zorder=4)
                ax.scatter(i+np.linspace(-.16,.16,10),data[data.split==split].sort_values('seed')[key],s=20,color=color,zorder=3)
            else:
                ax.errorbar(group.learning_rate,group[key+'_mean'],yerr=group[key+'_std'],fmt=fmt,color=color,
                            label=key.upper(),markersize=5,linewidth=1.8,elinewidth=1.2,capsize=4)
        if MODEL=='LC':
            ax.set_xticks([0,1],['RMSE','MAE']); ax.set_xlim(-.55,1.55)
        else:
            ax.set_xscale('log'); ax.xaxis.set_major_locator(FixedLocator(RATES))
            ax.xaxis.set_major_formatter(FixedFormatter(['1e-4','5e-4','1e-3','5e-3','1e-2']))
            ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlabel('Learning rate (log scale)'); ax.legend(frameon=False)
        ax.set_title(f'{title} (n={int(group.sample_count.iloc[0])})')
        ax.set_ylabel('Prediction error (m/s)'); ax.grid(True,alpha=.22,linewidth=.7)
        ax.spines[['top','right']].set_visible(False); ax.margins(x=.1,y=.15)
    fig.suptitle(f'{MODEL}: prediction performance across evaluation sets',fontsize=14,y=.98)
    fig.text(.5,.025,'Mean of 10 runs; error bars: ±1 sample SD (ddof = 1), not confidence intervals.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93],w_pad=2.2)
    fig.savefig(path,dpi=1200,facecolor='white'); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',type=Path,default=ROOT/RESULTS_DIRECTORY)
    p.add_argument('--heldout',type=Path,default=ROOT/'heldout_tracks_4_15')
    p.add_argument('--output',type=Path,default=ROOT/'evaluation_overview')
    args=p.parse_args()
    data,summary=load_data(args.results,args.heldout)
    args.output.mkdir(parents=True,exist_ok=True)
    data.to_csv(args.output/'all_run_metrics.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(args.output/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    draw(data,summary,args.output/'test_unseen_lhs_rmse_mae.png')
    files=[args.results/'all_run_metrics.csv',args.results/'experiment_config.json',args.heldout/'all_point_predictions.csv',args.heldout/'all_run_metrics.csv',args.heldout/'evaluation_config.json']
    cfg=dict(model=MODEL,inputs={str(f.resolve()):sha(f) for f in files},script_sha256=sha(__file__),
             panels=SPLITS,unseen_pooling='Concatenate 200+200 points per run, compute metrics, then average metrics over 10 runs',
             ddof=1,error_bars='sample standard deviation, not confidence intervals',dpi=1200,training_performed=False,
             interpretation='Seen trajectories have held-out test samples. LHS covers target domain; not all points necessarily extrapolate beyond training support.')
    (args.output/'plot_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['split','learning_rate','rmse_mean','rmse_std','mae_mean','mae_std']].to_string(index=False))
    print('FINISHED:',args.output.resolve())

if __name__=='__main__': main()
