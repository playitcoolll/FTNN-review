"""Compare source physics and five learned models on identical valid points."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from local_paths import INPUTS, method_paths, relative

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
MODELS = ['FTNN', 'DNN', 'RCNN', 'MFDNN', 'LC']
FEATURES = ['Hp_m', 'Mach', 'mass_kg', 'deltaT_degC', 'N1_pct']


def metrics(y, pred):
    e = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    assert len(e) > 0 and np.isfinite(e).all()
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(abs(e))))


def main():
    trajectories = pd.read_csv(OUT / 'trajectory_source_trim.csv')
    expected = pd.read_csv(OUT / 'evaluation_inputs.csv').iloc[:500]
    assert len(trajectories) == 500
    trajectories = trajectories.sort_values('RowID').reset_index(drop=True)
    assert trajectories.RowID.tolist() == list(range(1,501))
    assert not trajectories.Status.eq('not_run').any()
    assert np.allclose(trajectories[FEATURES], expected[FEATURES], atol=1e-9, rtol=1e-12)
    assert trajectories.Dataset.tolist() == expected.Dataset.tolist()
    assert np.allclose(trajectories.TargetPs, expected.TargetPs)
    assert np.array_equal(trajectories.Success.eq(1), np.isfinite(trajectories.Ps_mps))

    ft, ft_results, _ = method_paths('FTNN')
    lhs_target = pd.read_csv(ft_results / 'lhs_target_data_used.csv')
    lhs_file = next((INPUTS / 'lhs_source').glob('LHS_source_trim_Ps_*.xlsx'))
    lhs_source = pd.read_excel(lhs_file, sheet_name='Results')
    assert len(lhs_source) == len(lhs_target) == 600
    assert np.allclose(lhs_source[FEATURES], lhs_target[FEATURES], atol=1e-9, rtol=1e-12)
    lhs = lhs_source.copy()
    lhs['TargetPs'] = lhs_target.Ps_mps
    lhs['Dataset'] = 'lhs_target'
    lhs['RowID'] = np.arange(1, 601)
    lhs['Status'] = np.where(lhs.Success.eq(1), 'ok', 'source_trim_failed')

    datasets = {'test': trajectories[trajectories.Dataset.eq('test')].reset_index(drop=True),
                'unseen_tracks': trajectories[trajectories.Dataset.eq('unseen_tracks')].reset_index(drop=True),
                'lhs_target': lhs}
    rows, coverage, point_audits = [], [], []
    for split, data in datasets.items():
        valid = data.Success.eq(1) & np.isfinite(data.Ps_mps) & np.isfinite(data.TargetPs)
        assert valid.sum() > 0
        data['Included'] = valid
        coverage.append(dict(dataset=split, candidate_points=len(data), valid_points=int(valid.sum()),
                             failed_points=int((~valid).sum())))
        rows.append(dict(model='Source physics', learning_rate=np.nan, dataset=split, seed=np.nan,
                         n_points=int(valid.sum()), **metrics(data.loc[valid,'TargetPs'],data.loc[valid,'Ps_mps'])))
        point_audits.append(data)

    selected = {}
    for model in MODELS:
        root, result, heldout = method_paths(model)
        saved_metrics = pd.read_csv(result / 'all_run_metrics.csv')
        seedprefix = 'seed_' if model in ['DNN','MFDNN'] else 'source_'
        rate = None
        if model != 'LC':
            rate = float(saved_metrics[saved_metrics.split.eq('lhs_target')].groupby('learning_rate').rmse.mean().idxmin())
        selected[model] = rate
        unseen = pd.read_csv(heldout / 'all_point_predictions.csv')
        if rate is not None:
            unseen = unseen[unseen.learning_rate.eq(rate)]
        for seed in range(1000,1010):
            run = result / (seedprefix + str(seed))
            if rate is not None:
                run = run / ('lr_' + format(rate, 'g'))
            for split, data in datasets.items():
                if split == 'unseen_tracks':
                    predictions = unseen[unseen.seed.eq(seed)]
                    predictions = predictions.rename(columns={'track':'Track','t':'Time'})
                    joined = data[['Track','Time']].merge(predictions, on=['Track','Time'], how='left',
                                                        sort=False, validate='one_to_one')
                else:
                    joined = pd.read_csv(run / (split + '_predictions.csv'))
                if split == 'lhs_target':
                    features = joined[FEATURES]
                else:
                    features = joined[['h','mach','mass','dtem','N1']]
                assert len(joined) == len(data)
                assert np.allclose(features.to_numpy(), data[FEATURES].to_numpy(), atol=1e-9, rtol=1e-12)
                # Original decimal labels are used consistently for all methods.
                assert np.allclose(joined.ps_true, data.TargetPs, atol=1e-5, rtol=1e-7)
                valid = data.Included.to_numpy()
                rows.append(dict(model=model, learning_rate=rate, dataset=split, seed=seed,
                                 n_points=int(valid.sum()),
                                 **metrics(data.TargetPs.to_numpy()[valid],joined.ps_pred.to_numpy()[valid])))

    all_runs = pd.DataFrame(rows)
    summary_rows = []
    for split in datasets:
        for model in ['Source physics'] + MODELS:
            g = all_runs[all_runs.dataset.eq(split) & all_runs.model.eq(model)]
            assert len(g) == (1 if model == 'Source physics' else 10)
            assert g.n_points.nunique() == 1
            summary_rows.append(dict(model=model, learning_rate=g.learning_rate.iloc[0], dataset=split,
                                     n_points=int(g.n_points.iloc[0]), n_runs=len(g),
                                     rmse_mean=float(g.rmse.mean()), rmse_std=float(g.rmse.std(ddof=1)),
                                     mae_mean=float(g.mae.mean()), mae_std=float(g.mae.std(ddof=1))))
    summary = pd.DataFrame(summary_rows)
    coverage = pd.DataFrame(coverage)
    all_runs.to_csv(OUT / 'common_points_all_run_metrics.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT / 'comparison_with_source_physics.csv',index=False,encoding='utf-8-sig')
    coverage.to_csv(OUT / 'valid_point_counts.csv',index=False,encoding='utf-8-sig')
    pd.concat(point_audits,ignore_index=True).to_csv(OUT / 'point_validity_audit.csv',index=False,encoding='utf-8-sig')

    tex = [r'\begin{table}[width=\linewidth,cols=4,pos=htbp]',
           r'\caption{Prediction errors on common valid operating points following LHS-based learning-rate selection. Results for FTNN, DNN, RCNN, MFDNN, and LC are mean $\pm$ sample standard deviation over ten runs.}',
           r'\label{tab:selected-model-comparison}', r'\small',
           r'\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccc@{}}', r'\toprule',
           r'Method & Learning rate & RMSE (m/s) & MAE (m/s) \\']
    rate_tex = {model: '--' if rate is None else f'${rate:g}$' for model, rate in selected.items()}
    rate_tex['Source physics'] = '--'
    titles = {'test':'Held-out trajectory points', 'unseen_tracks':r'Pooled unseen trajectories: $4^\circ$ and $15^\circ$',
              'lhs_target':'Target-domain LHS validation subset'}
    for split in datasets:
        count = coverage[coverage.dataset.eq(split)].iloc[0]
        tex += [r'\midrule', r'\multicolumn{4}{l}{'+titles[split]+f' ($n={count.valid_points}$ of {count.candidate_points})'+r'} \\']
        for model in ['Source physics'] + MODELS:
            r = summary[summary.dataset.eq(split)&summary.model.eq(model)].iloc[0]
            name = 'Source physics' if model == 'Source physics' else model
            values = []
            for metric in ['rmse','mae']:
                v = f'{r[metric+"_mean"]:.4f}'
                if model != 'Source physics':
                    v += r'\pm'+f'{r[metric+"_std"]:.4f}'
                values.append('$'+v+'$')
            tex.append(name+' & '+rate_tex[model]+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular*}',r'\smallskip',r'\footnotesize',
            r'Only points with valid source-model trim solutions are included; identical point masks are applied to all methods. Counts show valid/candidate points. Source physics denotes the deterministic, uncorrected source simulation model. Learning rates remain those selected on the original 600-point target LHS set. Unseen-trajectory errors are pooled per run before averaging; SD uses $\mathrm{ddof}=1$.',
            r'\end{table}']
    text = '\n'.join(tex)+'\n'
    assert text.count('{') == text.count('}')
    (OUT / 'comparison_with_source_physics.tex').write_text(text,encoding='utf-8')
    audit = dict(selected_learning_rates=selected, selection_dataset_points=600,
                 comparison_mask='source trim success and finite source/target Ps',
                 source_model_repetitions=1, learned_model_repetitions=10,
                 target_labels='same original decimal target reference values for all methods',
                 reused_lhs_source_file=relative(lhs_file),
                 reused_lhs_sha256=hashlib.sha256(lhs_file.read_bytes()).hexdigest(),
                 trajectory_results_sha256=hashlib.sha256((OUT/'trajectory_source_trim.csv').read_bytes()).hexdigest())
    (OUT/'comparison_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(coverage.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
