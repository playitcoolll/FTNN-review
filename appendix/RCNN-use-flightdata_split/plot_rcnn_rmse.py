"""同图绘制五个学习率的验证/测试/LHS平均RMSE和MAE，误差条为±1个样本标准差。

运行：python plot_rcnn_rmse.py
可选：python plot_rcnn_rmse.py --show-runs  # 同时显示10条浅色单次运行曲线
输入：results_10sources_5lr_matched_seed/all_run_metrics.csv、experiment_config.json
输出：results_10sources_5lr_matched_seed/plots/rmse_mae_comparison.png（1200 DPI）、rmse_mae_summary.csv

先对每个模型分别计算RMSE和MAE，再分别对10次结果求算术平均。标准差ddof=1；不是
标准误或置信区间。不平均预测值，不重训模型，不自动选择学习率。
依赖：numpy、pandas、matplotlib。
"""

from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
SPLITS = ['validation','test','lhs_target']
TITLES = ['Flight validation','Flight test','Target LHS']
RATES = [1e-4,5e-4,1e-3,5e-3,1e-2]
SEEDS = list(range(1000,1010))
METRICS = ['rmse','mae']

def require(condition,message):
    if not condition:
        raise ValueError(message)

def load_and_summarize(results):
    config=json.loads((results/'experiment_config.json').read_text(encoding='utf-8-sig'))
    require(config['source_seeds']==SEEDS,'Expected source seeds 1000 through 1009')
    require(config.get('residual_seed_rule')=='source_seed','Expected source-matched residual seeds, not the previous fixed-1234 experiment')
    require(config['learning_rates']==RATES,'Expected the five configured learning rates')
    frame=pd.read_csv(results/'all_run_metrics.csv')
    needed={'source_seed','residual_template_seed','learning_rate','split','n','rmse','mae'}
    require(needed.issubset(frame.columns),f'Missing columns: {needed-set(frame.columns)}')
    require((frame.source_seed==frame.residual_template_seed).all(),'Residual seed does not match source seed')
    require(not frame.duplicated(['source_seed','learning_rate','split']).any(),'Duplicate run/learning-rate/split records')
    data=frame[frame.split.isin(SPLITS)].copy()
    require(len(data)==150,'Expected 10 sources x 5 learning rates x 3 datasets = 150 rows')
    require(set(data.learning_rate)==set(RATES),'Unexpected learning rate')
    require(set(data.source_seed)==set(SEEDS),'Unexpected source seed')
    require(np.isfinite(data[METRICS+['n']].to_numpy()).all() and (data[METRICS]>=0).all().all(),'Metric or count is invalid')
    require((data.mae<=data.rmse+1e-10).all(),'MAE exceeds RMSE for a run')
    if 'mse' in data:
        require(np.allclose(data.rmse**2,data.mse,rtol=1e-10,atol=1e-12),'RMSE squared does not match MSE')
    records=[]
    for split in SPLITS:
        for rate in RATES:
            group=data[(data.split==split)&(data.learning_rate==rate)]
            require(len(group)==10 and set(group.source_seed)==set(SEEDS),f'Incomplete group: {split}, {rate}')
            require((group.n==config['counts'][split]).all(),f'Inconsistent sample count: {split}')
            record=dict(split=split,learning_rate=rate,source_count=len(group),sample_count=int(group.n.iloc[0]))
            for metric in METRICS:
                values=group[metric].to_numpy(dtype=float)
                record.update({f'{metric}_mean':float(values.mean()),f'{metric}_std':float(values.std(ddof=1)),
                               f'{metric}_min':float(values.min()),f'{metric}_max':float(values.max())})
            records.append(record)
    summary=pd.DataFrame(records)
    # Independent check using pandas aggregation against the NumPy calculation.
    grouped=data.groupby(['split','learning_rate'])[METRICS].agg(['mean','std'])
    for row in summary.itertuples():
        reference=grouped.loc[(row.split,row.learning_rate)]
        for metric in METRICS:
            require(np.allclose([getattr(row,f'{metric}_mean'),getattr(row,f'{metric}_std')],reference[metric][['mean','std']].to_numpy(),rtol=1e-12,atol=1e-12),'Aggregation check failed')
    return data,summary

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results',type=Path,default=ROOT/'results_10sources_5lr_matched_seed')
    parser.add_argument('--output',type=Path,help='Default: <results>/plots')
    parser.add_argument('--show-runs',action='store_true',help='Overlay faint individual-source curves')
    args=parser.parse_args()
    data,summary=load_and_summarize(args.results)
    output=args.output or args.results/'plots'
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.titlesize':12,'axes.labelsize':11})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,split,title in zip(axes,SPLITS,TITLES):
        subset=summary[summary.split==split].sort_values('learning_rate')
        for metric,color,fmt in [('rmse','#166b96','o-'),('mae','#cb7024','s--')]:
            if args.show_runs:
                for seed in SEEDS:
                    single=data[(data.split==split)&(data.source_seed==seed)].sort_values('learning_rate')
                    ax.plot(single.learning_rate,single[metric],color=color,alpha=.15,linewidth=.7,zorder=1)
            ax.errorbar(subset.learning_rate,subset[f'{metric}_mean'],yerr=subset[f'{metric}_std'],
                        fmt=fmt,color=color,ecolor=color,markersize=5,label=metric.upper(),
                        linewidth=1.8,elinewidth=1.2,capsize=4,capthick=1.2,zorder=3)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(RATES))
        ax.xaxis.set_major_formatter(FixedFormatter(['1e-4','5e-4','1e-3','5e-3','1e-2']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Learning rate (log scale)')
        ax.set_ylabel('Prediction error (m/s)')
        ax.set_title(f'{title} (n={subset.sample_count.iloc[0]})')
        ax.grid(True,alpha=.22,linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.1,y=.15)
        ax.legend(frameon=False,loc='best')
    fig.suptitle('RCNN: mean RMSE and MAE across 10 paired source/residual seeds',fontsize=14,y=.98)
    fig.text(.5,.025,'Error bars: mean ± 1 sample SD (ddof = 1); not confidence intervals.',
             ha='center',fontsize=10,color='#444444')
    fig.tight_layout(rect=[0,.07,1,.93],w_pad=2.2)
    fig.savefig(output/'rmse_mae_comparison.png',dpi=1200,facecolor='white')
    plt.close(fig)
    summary.to_csv(output/'rmse_mae_summary.csv',index=False,encoding='utf-8-sig')
    config=dict(input=str((args.results/'all_run_metrics.csv').resolve()),
                input_sha256=hashlib.sha256((args.results/'all_run_metrics.csv').read_bytes()).hexdigest(),
                source_seeds=SEEDS,residual_seed_rule='source_seed',learning_rates=RATES,statistic='separate means of 10 individual-model RMSE and MAE values',
                error_bars='plus/minus sample standard deviation; ddof=1; not a confidence interval',
                show_runs=args.show_runs,output_format='png',dpi=1200,rows_checked=150,groups_checked=15,
                independent_aggregation_check='passed',numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__)
    (output/'rmse_mae_plot_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['split','learning_rate','source_count','rmse_mean','rmse_std','mae_mean','mae_std']].to_string(index=False))
    print(f'Figures and summary saved to: {output.resolve()}')

if __name__=='__main__':
    main()
