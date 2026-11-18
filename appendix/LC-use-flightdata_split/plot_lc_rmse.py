"""LC十个源模型的RMSE/MAE汇总：验证、测试、LHS三个子图。
运行：python plot_lc_rmse.py
柱高=10次指标的算术平均；误差条=±1个样本标准差(ddof=1)；散点=10次单独结果。
不平均预测值；LC没有学习率横轴。只输出1200 dpi PNG与汇总CSV。
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEEDS=list(range(1000,1010))
SPLITS=['validation','test','lhs_target']
TITLES=['Flight validation','Flight test','Target LHS']
METRICS=['rmse','mae']

def require(condition,message):
    if not condition: raise ValueError(message)


def load_and_summarize(results):
    config=json.loads((results/'experiment_config.json').read_text(encoding='utf-8-sig'))
    require(config.get('model')=='LC','Expected LC experiment')
    require(config['source_seeds']==SEEDS,'Expected source seeds 1000 through 1009')
    frame=pd.read_csv(results/'all_run_metrics.csv')
    needed={'source_seed','split','n','mse','rmse','mae','r2'}
    require(needed.issubset(frame.columns),f'Missing columns: {needed-set(frame.columns)}')
    require(len(frame)==40 and not frame.duplicated(['source_seed','split']).any(),'Expected 40 unique metric rows')
    require(set(frame.split)=={'train',*SPLITS},'Unexpected evaluation split')
    require(set(frame.source_seed)==set(SEEDS),'Unexpected source seeds')
    require(np.isfinite(frame[['n','mse','rmse','mae','r2']].to_numpy()).all(),'Invalid metrics')
    require((frame[['mse','rmse','mae']]>=0).all().all(),'Negative error metric')
    require((frame.mae<=frame.rmse+1e-10).all(),'MAE exceeds RMSE')
    require(np.allclose(frame.rmse**2,frame.mse,rtol=1e-10,atol=1e-12),'RMSE squared differs from MSE')
    rows=[]
    for split in ['train',*SPLITS]:
        group=frame[frame.split==split]
        require(len(group)==10 and set(group.source_seed)==set(SEEDS),f'Incomplete split: {split}')
        require((group.n==config['counts'][split]).all(),f'Wrong sample count: {split}')
        row=dict(split=split,seed_count=10,sample_count=int(group.n.iloc[0]))
        for key in ['mse','rmse','mae','r2']:
            values=group[key].to_numpy(dtype=float)
            row.update({f'{key}_mean':float(values.mean()),f'{key}_std':float(values.std(ddof=1)),
                        f'{key}_min':float(values.min()),f'{key}_max':float(values.max())})
            require(np.allclose([row[key+'_mean'],row[key+'_std']],[group[key].mean(),group[key].std(ddof=1)],rtol=1e-12,atol=1e-12),'Aggregation mismatch')
        rows.append(row)
    return frame,pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results',type=Path,default=ROOT/'results_10sources_lc')
    parser.add_argument('--output',type=Path,help='Default: <results>/plots')
    args=parser.parse_args()
    data,summary=load_and_summarize(args.results)
    output=args.output or args.results/'plots'; output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,split,title in zip(axes,SPLITS,TITLES):
        row=summary[summary.split==split].iloc[0]
        single=data[data.split==split].sort_values('source_seed')
        for i,(key,color) in enumerate(zip(METRICS,['#166b96','#cb7024'])):
            ax.bar(i,row[key+'_mean'],width=.5,color=color,alpha=.25,zorder=2)
            ax.errorbar(i,row[key+'_mean'],yerr=row[key+'_std'],fmt='D',color=color,
                        markersize=5,linewidth=1.5,capsize=5,zorder=4)
            ax.scatter(i+np.linspace(-.16,.16,10),single[key],s=23,color=color,
                       edgecolor='white',linewidth=.4,zorder=3)
        ax.set_xticks([0,1],['RMSE','MAE']); ax.set_xlim(-.55,1.55)
        ax.set_ylabel('Prediction error (m/s)'); ax.set_title(f'{title} (n={int(row.sample_count)})')
        ax.grid(axis='y',alpha=.22,linewidth=.7); ax.spines[['top','right']].set_visible(False)
        # Preserve the complete mean±SD error bar, even if its lower end is below zero.
        ax.margins(y=.15)
    fig.suptitle('LC: mean RMSE and MAE across 10 pretrained source models',fontsize=14,y=.98)
    fig.text(.5,.025,'Bars/diamonds: mean; error bars: ±1 sample SD (ddof = 1); dots: individual runs.',
             ha='center',fontsize=10,color='#444444')
    fig.tight_layout(rect=[0,.07,1,.93],w_pad=2.2)
    fig.savefig(output/'rmse_mae_comparison.png',dpi=1200,facecolor='white'); plt.close(fig)
    summary.to_csv(output/'rmse_mae_summary.csv',index=False,encoding='utf-8-sig')
    config=dict(model='LC',source_seeds=SEEDS,input=str((args.results/'all_run_metrics.csv').resolve()),
                input_sha256=hashlib.sha256((args.results/'all_run_metrics.csv').read_bytes()).hexdigest(),
                statistic='mean of 10 individual RMSE/MAE values, not metrics of ensemble-averaged predictions',
                error_bars='plus/minus sample SD; ddof=1; not confidence intervals',output_format='png',dpi=1200,
                numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__)
    (output/'rmse_mae_plot_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['split','seed_count','rmse_mean','rmse_std','mae_mean','mae_std']].to_string(index=False))
    print(f'Figures and summary saved to: {output.resolve()}')

if __name__=='__main__':
    main()
