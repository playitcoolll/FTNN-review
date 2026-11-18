# 源动力学模型配平基线

[中文](#中文) | [English](#english)

## 中文

本目录将未修正的源动力学模型与 FTNN、DNN、RCNN、MFDNN、LC 在相同工况下比较。
所有误差均以目标模型的 Ps 为参考值；源模型配平得到的 Ps 是待评价的基线预测。

本次计算已完成：轨迹测试集 97/100 点有效，未见轨迹 400/400 点有效，LHS 566/600 点有效。
重新计算的 500 个轨迹点中，RowID 90、92、99 经三组初值尝试后仍未满足收敛残差要求，已排除。
它们均属于 25° 轨迹，对应时间分别为 2109.69、2189.69、2009.69。
单点复核与已有源 LHS Ps 的差为约 4.3e-14 m/s。

### 数据与计算

- `evaluation_inputs.csv`：前 100 行为轨迹测试点，接下来 400 行为未见 4°、15° 轨迹点。
  第 501 行仅为已有源 LHS 配平结果的复核点，不纳入比较。
- `trajectory_source_trim.csv`：已缓存的 500 个轨迹点的源模型配平结果，对比脚本直接读取。
- LHS 源模型结果复用 `data/inputfile/lhs_source/LHS_source_trim_Ps_*.xlsx`，
  并核对全部 600 个输入工况与目标 LHS 一致。

### 排除规则

未收敛、残差超限、无效数值或配平输入触限的点不计入误差。
其 `Ps_mps` 为缺失值，候选解保留在 `Ps_candidate_mps` 中供检查，不能用于评价。
源模型失败点对应的五种方法预测也从本目录比较表中排除，以保持评价样本一致。

### 汇总

本目录只保留对比汇总脚本，配平计算代码已移除，直接使用已缓存的配平结果。

- `summarize_common_points.py`：在相同有效点上计算源物理基线和五种方法的 RMSE/MAE，
  生成 `comparison_with_source_physics.csv/.tex`。
- `build_full_comparison.py`：从五种方法的各自结果目录里选学习率、汇总完整评估集的指标。
- `restore_full_evaluation_table.py`：把源物理基线和五种方法合并，
  生成 `comparison_full_sets_with_source_baseline.csv/.tex`。

- `comparison_with_source_physics.tex`：可插入论文的完整 LaTeX 表格。
- `comparison_with_source_physics.csv`：完整比较表，包含有效样本数、运行次数、均值和标准差。
- `common_points_all_run_metrics.csv`：每个方法、每个种子的单次误差。
- `valid_point_counts.csv`：三个数据集的候选点数、有效点数、排除点数。
- `point_validity_audit.csv`：全部候选点的纳入标记及配平信息。
- `comparison_manifest.json`：学习率、数据来源和结果文件哈希。

学习率仍按原始 600 点目标 LHS 集上的十次平均 RMSE 最小值选择：
FTNN 为 0.001，DNN 为 0.0001，RCNN 为 0.01，MFDNN 为 0.005。
本次不根据排除后的子集重新选学习率，也不重新训练模型。
原学习率敏感性图使用完整数据集；本目录比较表使用共同有效点，须在表注中说明。

源动力学模型是确定性模型，报告一组误差，不人为复制成十次结果或标注零标准差。
其余五种方法先分别计算每次运行的 RMSE、MAE，再统计十次均值和样本标准差（ddof=1）。
未见轨迹先在每次运行内合并两条轨迹，再计算误差。

### 使用方法

本目录的对比脚本不重新训练模型，也不重新计算源模型配平；它们只把已有的配平结果与五种方法的结果汇总成表。运行前需要先具备以下结果：

- 五种方法各自的训练与独立轨迹评估结果（位于各自方法目录的默认路径）：

  | 方法 | 训练脚本 | 默认结果目录 |
  | --- | --- | --- |
  | FTNN | `run_ftnn_50.py` | `results_10sources_5lr` |
  | DNN | `run_dnn_50.py` | `results_10seeds_5lr` |
  | RCNN | `run_rcnn_50.py` | `results_10sources_5lr_matched_seed` |
  | LC | `run_lc_10.py` | `results_10sources_lc` |
  | MFDNN | `run_mfdnn_50.py` | `results_10seeds_5lr` |

  每个方法目录还需运行 `evaluate_tracks_4_15.py`，生成 `heldout_tracks_4_15/all_point_predictions.csv`；
  FTNN 的结果目录还需包含 `lhs_target_data_used.csv`。这些结果目录被 `.gitignore` 排除、不在仓库中，
  需先按根目录 README 的步骤在各自方法目录内跑完训练和评估。

- 本目录已缓存的源物理配平结果：`trajectory_source_trim.csv`、`evaluation_inputs.csv`，
  以及 `data/inputfile/lhs_source/LHS_source_trim_Ps_*.xlsx`。

在 `source_physics_baseline` 目录内按顺序执行：

```powershell
& $python summarize_common_points.py
& $python restore_full_evaluation_table.py
```

`summarize_common_points.py` 生成共同有效点比较表 `comparison_with_source_physics.csv/.tex`，以及
`common_points_all_run_metrics.csv`、`valid_point_counts.csv`、`point_validity_audit.csv`、`comparison_manifest.json`。
`restore_full_evaluation_table.py` 内部会调用 `build_full_comparison.py`，生成完整评估集对比表
`comparison_full_sets_with_source_baseline.csv/.tex` 和根目录的 `selected_learning_rates_comparison.csv`。
也可以单独运行 `python build_full_comparison.py`，只重新生成学习率选择表。

`trajectory_source_trim.mat` 保存详细工作点、求解报告及配平配置。

---

## English

This directory compares the uncorrected source dynamics model with FTNN, DNN, RCNN, MFDNN, and LC under identical operating conditions.
All errors use the target model's Ps as the reference; the source model's trimmed Ps is the baseline prediction being evaluated.

The computation is already complete: 97/100 trajectory test points, 400/400 unseen-trajectory points, and 566/600 LHS points are valid.
Among the 500 recomputed trajectory points, RowID 90, 92, and 99 failed to satisfy the convergence-residual requirements after three sets of initial guesses and were excluded.
They all belong to the 25° trajectory, with times 2109.69, 2189.69, and 2009.69, respectively.
The single-point check differs from the existing source LHS Ps by about 4.3e-14 m/s.

### Data and computation

- `evaluation_inputs.csv`: the first 100 rows are trajectory test points, and the next 400 rows are the unseen 4° and 15° trajectory points.
  Row 501 is only a verification point for the existing source LHS trim result and is not included in the comparison.
- `trajectory_source_trim.csv`: cached source-model trim results for the 500 trajectory points, read directly by the comparison scripts.
- The source LHS results reuse `data/inputfile/lhs_source/LHS_source_trim_Ps_*.xlsx`,
  and all 600 input conditions are checked to match the target LHS.

### Exclusion rules

Points that do not converge, exceed residual limits, contain invalid values, or reach trim-input limits are excluded from the errors.
Their `Ps_mps` is missing; candidate solutions are kept in `Ps_candidate_mps` for inspection and cannot be used for evaluation.
Predictions of the five methods at source-model failure points are also excluded from this directory's comparison table to keep the evaluation sample consistent.

### Aggregation

This directory keeps only the comparison/aggregation scripts; the trim computation code has been removed and the cached trim results are used directly.

- `summarize_common_points.py`: computes RMSE/MAE for the source physics baseline and the five methods on identical valid points,
  producing `comparison_with_source_physics.csv/.tex`.
- `build_full_comparison.py`: selects learning rates from each method's result directory and aggregates the full evaluation-set metrics.
- `restore_full_evaluation_table.py`: merges the source physics baseline with the five methods,
  producing `comparison_full_sets_with_source_baseline.csv/.tex`.

- `comparison_with_source_physics.tex`: a complete LaTeX table ready to insert into the paper.
- `comparison_with_source_physics.csv`: the complete comparison table with valid sample sizes, run counts, means, and standard deviations.
- `common_points_all_run_metrics.csv`: per-run errors for each method and seed.
- `valid_point_counts.csv`: candidate, valid, and excluded point counts for the three datasets.
- `point_validity_audit.csv`: inclusion flags and trim information for all candidate points.
- `comparison_manifest.json`: learning rates, data sources, and result-file hashes.

Learning rates are still selected by the minimum mean LHS RMSE over ten runs on the original 600-point target LHS set:
FTNN 0.001, DNN 0.0001, RCNN 0.01, MFDNN 0.005.
Learning rates are not reselected on the subset after exclusions, and models are not retrained.
The original learning-rate sensitivity figures use the full dataset; this directory's comparison table uses common valid points, which must be noted in the table caption.

The source dynamics model is deterministic and reports a single set of errors; it is not artificially replicated into ten runs or assigned a zero standard deviation.
For the other five methods, RMSE and MAE are computed for each run first, then the mean and sample standard deviation over ten runs are reported (ddof=1).
Unseen trajectories are pooled across the two trajectories within each run before computing errors.

### Usage

The comparison scripts here do not retrain models or recompute the source-model trim; they only aggregate the cached trim results and the five methods' results into tables. Before running, the following results must already exist:

- Training and unseen-trajectory evaluation results for each of the five methods (at their default locations):

  | Method | Training script | Default results directory |
  | --- | --- | --- |
  | FTNN | `run_ftnn_50.py` | `results_10sources_5lr` |
  | DNN | `run_dnn_50.py` | `results_10seeds_5lr` |
  | RCNN | `run_rcnn_50.py` | `results_10sources_5lr_matched_seed` |
  | LC | `run_lc_10.py` | `results_10sources_lc` |
  | MFDNN | `run_mfdnn_50.py` | `results_10seeds_5lr` |

  Each method directory must also have `heldout_tracks_4_15/all_point_predictions.csv` (from `evaluate_tracks_4_15.py`);
  the FTNN results directory must also contain `lhs_target_data_used.csv`. These result directories are gitignored and not in
  the repository, so run training and evaluation in each method directory first, following the root README.

- The source-physics trim results cached in this directory: `trajectory_source_trim.csv`, `evaluation_inputs.csv`,
  and `data/inputfile/lhs_source/LHS_source_trim_Ps_*.xlsx`.

Run the following in order from `source_physics_baseline`:

```powershell
& $python summarize_common_points.py
& $python restore_full_evaluation_table.py
```

`summarize_common_points.py` produces the common-valid-point comparison table `comparison_with_source_physics.csv/.tex`, plus
`common_points_all_run_metrics.csv`, `valid_point_counts.csv`, `point_validity_audit.csv`, and `comparison_manifest.json`.
`restore_full_evaluation_table.py` calls `build_full_comparison.py` internally and produces the full evaluation-set comparison table
`comparison_full_sets_with_source_baseline.csv/.tex` and the root-level `selected_learning_rates_comparison.csv`.
You can also run `python build_full_comparison.py` alone to regenerate only the learning-rate selection table.

`trajectory_source_trim.mat` stores detailed operating points, solver reports, and trim configuration.
