"""Keep full evaluation sets for five methods; exclude failed trims only for source physics."""
from pathlib import Path
import pandas as pd
from build_full_comparison import build

OUT = Path(__file__).resolve().parent
full = build()
source = pd.read_csv(OUT / 'comparison_with_source_physics.csv')
source = source[source.model.eq('Source physics')].copy()
source['evaluation_scope'] = 'successful source-model trim points only'
full['evaluation_scope'] = 'full evaluation set'
models = ['Source physics', 'FTNN', 'DNN', 'RCNN', 'MFDNN', 'LC']
splits = ['test', 'unseen_tracks', 'lhs_target']
merged = pd.concat([source, full], ignore_index=True)
ordered = pd.concat([merged[merged.dataset.eq(s) & merged.model.eq(m)]
                     for s in splits for m in models], ignore_index=True)
assert len(ordered) == 18
for split, n in zip(splits, [100, 400, 600]):
    assert ordered[ordered.dataset.eq(split) & ordered.model.ne('Source physics')].n_points.eq(n).all()
ordered.to_csv(OUT / 'comparison_full_sets_with_source_baseline.csv', index=False, encoding='utf-8-sig')

tex = [r'\begin{table}[width=\linewidth,cols=4,pos=htbp]',
       r'\caption{Prediction errors of the source model and five comparison methods at the selected learning rates. Values for FTNN, DNN, RCNN, MFDNN, and LC are mean $\pm$ sample standard deviation over ten runs.}',
       r'\label{tab:selected-model-comparison}', r'\small',
       r'\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccc@{}}',
       r'\toprule', r'Method & Learning rate & RMSE (m/s) & MAE (m/s) \\']
rates = {row.model: '--' if pd.isna(row.learning_rate) else f'${row.learning_rate:g}$' for _, row in ordered.iterrows()}
titles = dict(test='Held-out trajectory points', unseen_tracks='Pooled unseen trajectories',
              lhs_target='Target-domain LHS validation set')
for split in splits:
    tex.extend([r'\midrule', r'\multicolumn{4}{l}{'+titles[split]+r'} \\'])
    for _, row in ordered[ordered.dataset.eq(split)].iterrows():
        baseline = row.model == 'Source physics'
        values = []
        for metric in ['rmse','mae']:
            val = f'{row[metric+"_mean"]:.4f}'
            if not baseline:
                val += r'\pm'+f'{row[metric+"_std"]:.4f}'
            values.append('$'+val+'$')
        name = 'Source model' if baseline else row.model
        tex.append(name+' & '+rates[row.model]+' & '+' & '.join(values)+r' \\')
tex.extend([r'\bottomrule', r'\end{tabular*}', r'\smallskip', r'\footnotesize',
            r'Source model denotes the uncorrected source dynamics model, whose deterministic errors are reported without standard deviations. Trajectory errors use the original trajectory labels; LHS errors use target-model trim labels. The five comparison methods are evaluated on the full evaluation sets; only source-model errors exclude points with unsuccessful trim solutions. Learning rates are selected using the full LHS validation set. Unseen-trajectory errors are pooled per run before averaging across runs. Sample standard deviations use $\mathrm{ddof}=1$.',
            r'\end{table}'])
text = '\n'.join(tex)+'\n'
assert text.count('{') == text.count('}')
(OUT/'comparison_full_sets_with_source_baseline.tex').write_text(text,encoding='utf-8')
print(ordered[['model','dataset','n_points','rmse_mean','mae_mean']].to_string(index=False))
