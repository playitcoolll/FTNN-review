# 超参数选择与公平比较

[中文](#中文) | [English](#english)

## 中文

### 1. 对应论文

本目录对应论文 *A Transfer Learning Neural Network–Based Method for Aircraft Specific Excess Power Model Identification* 的 **第 4.2.1 节：Learning-Rate Sensitivity and Model Comparison**。在所提供的 44 页稿件中，该节位于第 20–23 页，主要结果为 **图 8（第 22 页）和表 13（第 23 页）**。页码和编号以该稿件为准。

| 内容| 论文对应|
| --- | --- |
| FTNN、DNN、RCNN、MFDNN 的五学习率、十次重复实验| 第 4.2.1 节；图 8|
| LC 线性校正| 公式（16）；表 13|
| RCNN 残差校正| 公式（17）|
| MFDNN 联合多保真训练| 公式（18）–（20）|
| 固定轨迹划分| 第 4.1.2 节末尾，第 20 页|
| 未见轨迹 4°、15° 评估| 图 8、表 13 的 pooled unseen trajectories|

**范围说明：**这里的 FTNN 为全层微调，相当于后续第 4.2.2 节中的 A1。A2/A3 层冻结、训练样本量、轨迹组合及任务性能实验不在本目录的五组主程序中。表 13 的 Source model 指未经校正的源动力学模型，不等同于预训练源神经网络；这条动力学基线的计算不是这五组主程序的功能。

本仓库开源核心算法代码。论文其余部分的统计分析代码（FTNN 层冻结消融、训练样本量、轨迹组合、任务分析，以及源/目标域散点曲面、Ps 切片、运动学与动力学 Ps 对比、数据对比等）未包含在本仓库中。

代码生成逐次运行指标、统计结果及单方法图片。训练脚本不自动选择最优学习率，也不直接生成完整的论文表 13 或图 8 的跨方法组合排版。

### 2. 实验设置

| 项目| 设置|
| --- | --- |
| 训练轨迹| 0°、2.5°、6°、12°、25°（T1、T2、T4、T5、T7） |
| 固定划分| 每条 140/40/20 点；合计训练 700、验证 200、测试 100|
| 未见轨迹| 4°、15°（T3、T6）；每条合并三个划分共 200 点，合计 400 点|
| LHS | 论文使用 600 点目标域 LHS 验证集|
| 学习率| `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2` |
| 重复| 每个学习率 10 次；种子 1000–1009|
| 优化| 全批量 Adam，默认 3000 轮，无早停，保留最终轮参数|
| FTNN | 10 个源网络；微调种子固定 1234；全层更新|
| RCNN | 源网络冻结；残差种子与源种子匹配|
| LC | 每个源网络拟合一次最小二乘线性校正，共 10 次；无需学习率|

LHS 用于选择学习率，不能称为独立测试集。运行时程序会筛选 LHS 中成功且数值有效的记录；替换数据后不要默认仍有 600 个有效点。

### 3. 文件结构和依赖

```text
FTNN-review/
├── README.md
├── DNN-use-flightdata_split/
├── FTNN-use-flightdata_split/
├── RCNN-use-flightdata_split/
├── LC-use-flightdata_split/
├── MFDNN-use-flightdata_split/
└── data/
    ├── inputfile/
    │   ├── flightdata/flightdata_split/        # 21 个轨迹 Excel 文件
    │   └── lhs_target/LHS_target_trim_Ps_20261028_203140_904.xlsx
    └── MSFS747-MODEL/
        ├── flightdata/out_747-800_3.xlsx       # MFDNN 低保真数据
        └── out_pcnn_5D/
            ├── out_1000/
            ├── ...
            └── out_1009/
```

以上列出五种方法实际需要的主要输入；“data”中的其他文件可作为附带资料保留。每套源模型必须包含 `run_config.json`、`normalization_params.npz`、`pcnn_model.ckpt.index` 和全部 `pcnn_model.ckpt.data-*` 分片。FTNN、RCNN、LC 使用这些源模型；DNN 不加载源模型；MFDNN 使用低保真 Excel 联合训练。

默认输入路径以脚本位置为基准，统一指向本目录中的“data”，无需外部仓库或 `portable_paths.py`。整体移动本文件夹时保持内部结构即可。评估和绘图还需要本地训练产生的结果。显式传入命令行路径参数会覆盖默认值。

### 4. Python 环境

已验证复现环境为 **Windows CPU、Python 3.9.7、TensorFlow 2.15.1、NumPy 1.26.4、pandas 2.3.2、openpyxl 3.1.5、matplotlib 3.9.4**。请先安装 Python 3.9.7，再在本目录执行以下 PowerShell 命令。完整依赖版本见 `requirements-reproduction.txt`。不要把仅能运行代码的其他版本视为数值复现环境。

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

后续命令使用 `$python`，避免 PowerShell 激活脚本策略影响操作。若使用现有环境，可改为 `$python = (Get-Command python).Source`。训练脚本设置为使用 CPU，不要求 GPU。

### 5. 运行步骤

#### 5.1 单方法运行：以 FTNN 为例

从本目录开始执行。先检查输入，再训练；训练完成后进行独立轨迹评估，最后绘图。

```powershell
Push-Location "FTNN-use-flightdata_split"
& $python run_ftnn_50.py --check-only
# 检查成功后再继续。
& $python run_ftnn_50.py
& $python evaluate_tracks_4_15.py
& $python plot_ftnn_rmse.py
& $python plot_evaluation_overview.py
Pop-Location
```

`--check-only` 不进行优化更新、不生成训练结果，但会加载数据及模型并检查预测接口，因此仍需 TensorFlow。它不能代替正式训练。下表列出各方法可替换的脚本名；评估和总览脚本名在各目录中相同。

| 方法| 主程序| 单方法误差图| 默认结果目录|
| --- | --- | --- | --- |
| DNN | `run_dnn_50.py` | `plot_dnn_rmse.py` | `results_10seeds_5lr` |
| FTNN | `run_ftnn_50.py` | `plot_ftnn_rmse.py` | `results_10sources_5lr` |
| RCNN | `run_rcnn_50.py` | `plot_rcnn_rmse.py` | `results_10sources_5lr_matched_seed` |
| LC | `run_lc_10.py` | `plot_lc_rmse.py` | `results_10sources_lc` |
| MFDNN | `run_mfdnn_50.py` | `plot_mfdnn_rmse.py` | `results_10seeds_5lr` |

#### 5.2 顺序运行全部方法

下面命令从本目录执行，先检查、再运行每种方法；任一步失败就停止。四种神经网络共 200 次训练，另外有 10 次 LC 拟合，完成时间取决于机器性能。

```powershell
$methods = @('dnn', 'ftnn', 'rcnn', 'lc', 'mfdnn')
foreach ($method in $methods) {
    $folder = $method.ToUpper() + '-use-flightdata_split'
    $count = if ($method -eq 'lc') { 10 } else { 50 }
    Push-Location $folder
    try {
        & $python "run_${method}_${count}.py" --check-only
        if ($LASTEXITCODE -ne 0) { throw "$method input check failed" }
        & $python "run_${method}_${count}.py"
        if ($LASTEXITCODE -ne 0) { throw "$method training failed" }
        & $python evaluate_tracks_4_15.py
        if ($LASTEXITCODE -ne 0) { throw "$method evaluation failed" }
        & $python "plot_${method}_rmse.py"
        if ($LASTEXITCODE -ne 0) { throw "$method error plot failed" }
        & $python plot_evaluation_overview.py
        if ($LASTEXITCODE -ne 0) { throw "$method overview failed" }
    } finally {
        Pop-Location
    }
}
```

#### 5.3 中断恢复和参数

在对应算法目录内，例如：

```powershell
& $python run_ftnn_50.py --resume
& $python run_ftnn_50.py --help
& $python evaluate_tracks_4_15.py --help
```

`--resume` 跳过已完整完成且通过文件校验的运行，不代表从未完成运行的最后一个 epoch 恢复。请保留完整结果目录。搬迁已有结果后，旧清单中的绝对路径可能导致恢复检查失败；不要手工删除校验信息，可保留原结果并用新的 `--output` 目录重新运行。

| 参数| 适用范围|
| --- | --- |
| `--data-root` | 训练及独立轨迹评估的轨迹目录|
| `--lhs-file` | 五种方法的目标域 LHS 文件|
| `--source-root` | FTNN、RCNN、LC 的源模型根目录|
| `--low-file` | MFDNN 低保真 Excel|
| `--epochs` | 四种神经网络的轮数；LC 不适用|
| `--results` | 评估及绘图的训练结果目录|
| `--heldout` | 总览绘图读取的独立轨迹评估目录|
| `--output` | 当前脚本的输出目录|

如果训练时更改 `--output`，后续脚本必须通过 `--results` 指向同一个目录。如果独立轨迹评估更改 `--output`，总览绘图需要相应更改 `--heldout`。复现论文应保留默认 3000 轮及原始数据。

### 6. 输出与论文结果整理

所有输出默认位于对应算法目录内。

| 输出| 内容|
| --- | --- |
| `<results>/experiment_config.json` | 实验设置及输入审计|
| `<results>/all_run_metrics.csv` | 各运行的指标，尚未跨种子平均|
| `<results>/source_<seed>/lr_<rate>/` | FTNN、RCNN 每次运行|
| `<results>/seed_<seed>/lr_<rate>/` | DNN、MFDNN 每次运行|
| `<results>/source_<seed>/` | LC 每次拟合|
| `<results>/plots/` | `rmse_mae_comparison.png`、`rmse_mae_summary.csv` 等|
| `heldout_tracks_4_15/` | 独立轨迹逐点预测、指标及 `evaluation_config.json`|
| `evaluation_overview/` | 三类评估集合的总览图、`all_run_metrics.csv` 和 `metrics_summary.csv`|

整理图 8、表 13 时遵循以下顺序：

1. 每种神经网络分别计算每个学习率在 LHS 上十次运行的平均 RMSE，选择均值最低者。
2. 将选定学习率用于该方法全部评估集合，不能为不同集合分别选择学习率。
3. 每次运行内先合并 4°、15° 的逐点预测并计算误差，再跨十次运行计算均值和样本标准差（`ddof=1`）；不要直接平均两个轨迹的 RMSE。
4. LC 汇总十次拟合，不参与学习率搜索。源动力学基线需要另行取得。

论文表 13 报告的学习率为 FTNN `1e-3`、DNN `1e-4`、RCNN `1e-2`、MFDNN `5e-3`。这些是稿件报告值，应与本地重新运行的汇总核对，而不是强制指定为运行结果。这里 FTNN 的 LHS RMSE 报告值为 `3.25 ± 0.53 m/s`；摘要中的 `2.12 ± 0.22 m/s` 属于后续 A3 层冻结实验。

### 7. 验证范围

2026-11-16 已完成全部五种方法的训练和评估，并对 FTNN、MFDNN 在上述原版本环境中各重新运行 50 次、每次 3000 轮。两种方法的全部训练历史、评估指标（不含耗时）、测试集/LHS 逐点预测和未见轨迹预测与原始实验结果数值完全一致，已比较数值的最大绝对差为 0。

FTNN 的 LHS RMSE 为 `3.256400 ± 0.534217 m/s`，MFDNN 为 `6.928684 ± 3.319786 m/s`；按论文截断规则显示为 `3.25 ± 0.53` 和 `6.92 ± 3.31`。所选学习率分别为 `0.001` 和 `0.005`。DNN、RCNN、LC 的完整运行结果也与论文显示精度一致，但本次未在原版本环境中重复这三种方法。

Python 3.12 / TensorFlow 2.21 / NumPy 2.5 环境曾产生 FTNN、MFDNN 数值差异；恢复上述原版本环境后差异消失。此验证针对本机 CPU 和已提供的代码、数据、源权重，不保证任意平台或依赖版本都逐位一致。

---

## English

### Hyperparameter Selection and Fair Comparison

### 1. Connection to the paper

This directory implements **Section 4.2.1, Learning-Rate Sensitivity and Model Comparison**, of *A Transfer Learning Neural Network–Based Method for Aircraft Specific Excess Power Model Identification*. In the supplied 44-page manuscript, this section spans pages 20–23; its main results are **Figure 8 (page 22) and Table 13 (page 23)**. These references apply to that manuscript version.

|Component |Paper correspondence |
| --- | --- |
|FTNN, DNN, RCNN, and MFDNN: five learning rates and ten repetitions |Section 4.2.1; Figure 8 |
|Linear correction |Equation (16); Table 13 |
|Residual correction |Equation (17) |
|Joint multi-fidelity training |Equations (18)–(20) |
|Fixed trajectory partition |End of Section 4.1.2, page 20 |
|Evaluation on unseen 4° and 15° trajectories |Pooled unseen trajectories in Figure 8 and Table 13 |

**Scope:** FTNN here updates all layers and corresponds to A1 in Section 4.2.2. The five main programs do not implement the A2/A3 freezing experiments, training-size study, trajectory-combination study, or task-level evaluation. The “Source model” row of Table 13 refers to the uncorrected source dynamics model, not the pretrained neural network; computing that dynamics baseline is outside these five programs.

This repository open-sources the core algorithm code. The statistical-analysis code for the remaining parts of the paper (the FTNN layer-freezing ablation, training-size study, trajectory combinations, task analysis, source/target-domain scatter and surface plots, Ps slices, kinematic-vs-dynamic Ps comparison, data comparison, and related analyses) is not included in this repository.

The scripts produce per-run metrics, summaries, and per-method figures. Training does not automatically select a learning rate or assemble the complete publication layout of Table 13 or the cross-method Figure 8.

### 2. Experimental settings

|Item |Setting |
| --- | --- |
|Training trajectories | 0°, 2.5°, 6°, 12°, 25° (T1, T2, T4, T5, T7) |
|Fixed split |140/40/20 points per trajectory; 700 training, 200 validation, 100 test points |
|Unseen trajectories |4° and 15° (T3, T6); three splits pooled into 200 points per trajectory, 400 total |
| LHS |The paper uses a 600-point target-domain LHS validation set |
|Learning rates | `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2` |
|Repetitions |Ten per learning rate; seeds 1000–1009 |
|Optimization |Full-batch Adam, 3000 epochs by default, no early stopping, final-epoch parameters |
| FTNN |Ten source networks; fine-tuning seed 1234; all layers updated |
| RCNN |Frozen source; residual seed matched to source seed |
| LC |One least-squares correction per source network, ten fits; no learning rate |

LHS is used for learning-rate selection and is not an independent test set. The loader retains successful, finite LHS records; replacing the input file does not guarantee 600 usable points.

### 3. Layout and dependencies

```text
FTNN-review/
├── README.md
├── DNN-use-flightdata_split/
├── FTNN-use-flightdata_split/
├── RCNN-use-flightdata_split/
├── LC-use-flightdata_split/
├── MFDNN-use-flightdata_split/
└── data/
    ├── inputfile/
    │   ├── flightdata/flightdata_split/        # 21 trajectory Excel files
    │   └── lhs_target/LHS_target_trim_Ps_20261028_203140_904.xlsx
    └── MSFS747-MODEL/
        ├── flightdata/out_747-800_3.xlsx       # MFDNN low-fidelity data
        └── out_pcnn_5D/
            ├── out_1000/
            ├── ...
            └── out_1009/
```

The tree lists the main inputs required by the five methods; other bundled files may remain as supporting material. Each source-model directory needs `run_config.json`, `normalization_params.npz`, `pcnn_model.ckpt.index`, and all `pcnn_model.ckpt.data-*` shards. FTNN, RCNN, and LC use these source models; DNN does not load a source model; MFDNN jointly trains using the low-fidelity Excel data.

Default input paths are resolved from the script location into the bundled dependency directory. No external repository or `portable_paths.py` is required. Keep the internal layout when moving the whole folder. Evaluation and plotting also require locally generated training results. Explicit command-line paths override the defaults.

### 4. Python environment

The verified reproduction environment is **Windows CPU, Python 3.9.7, TensorFlow 2.15.1, NumPy 1.26.4, pandas 2.3.2, openpyxl 3.1.5, and matplotlib 3.9.4**. Install Python 3.9.7 first, then run these PowerShell commands from this directory. All installed dependency versions are pinned in `requirements-reproduction.txt`. Compatibility with another version does not establish numerical reproduction.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

The commands below use `$python` to avoid relying on PowerShell activation scripts. For an existing environment, use `$python = (Get-Command python).Source`. Training scripts select CPU execution; a GPU is not required.

### 5. Running the experiments

#### 5.1 One method: FTNN example

Start from this directory. Check inputs first, train, evaluate unseen trajectories after training finishes, and then generate plots.

```powershell
Push-Location "FTNN-use-flightdata_split"
& $python run_ftnn_50.py --check-only
# Continue only after the check succeeds.
& $python run_ftnn_50.py
& $python evaluate_tracks_4_15.py
& $python plot_ftnn_rmse.py
& $python plot_evaluation_overview.py
Pop-Location
```

`--check-only` performs no optimizer updates and writes no training results, but loads data/models and checks prediction interfaces, so TensorFlow is still required. It does not replace training. Substitute the method-specific filenames below; evaluation and overview script names are identical across directories.

|Method |Main program |Error plot |Default results directory |
| --- | --- | --- | --- |
| DNN | `run_dnn_50.py` | `plot_dnn_rmse.py` | `results_10seeds_5lr` |
| FTNN | `run_ftnn_50.py` | `plot_ftnn_rmse.py` | `results_10sources_5lr` |
| RCNN | `run_rcnn_50.py` | `plot_rcnn_rmse.py` | `results_10sources_5lr_matched_seed` |
| LC | `run_lc_10.py` | `plot_lc_rmse.py` | `results_10sources_lc` |
| MFDNN | `run_mfdnn_50.py` | `plot_mfdnn_rmse.py` | `results_10seeds_5lr` |

#### 5.2 Run all methods sequentially

Run this block from this directory. It checks and runs each method in sequence and stops at the first failure. The four neural-network methods perform 200 training runs in total, plus ten LC fits; runtime depends on the machine.

```powershell
$methods = @('dnn', 'ftnn', 'rcnn', 'lc', 'mfdnn')
foreach ($method in $methods) {
    $folder = $method.ToUpper() + '-use-flightdata_split'
    $count = if ($method -eq 'lc') { 10 } else { 50 }
    Push-Location $folder
    try {
        & $python "run_${method}_${count}.py" --check-only
        if ($LASTEXITCODE -ne 0) { throw "$method input check failed" }
        & $python "run_${method}_${count}.py"
        if ($LASTEXITCODE -ne 0) { throw "$method training failed" }
        & $python evaluate_tracks_4_15.py
        if ($LASTEXITCODE -ne 0) { throw "$method evaluation failed" }
        & $python "plot_${method}_rmse.py"
        if ($LASTEXITCODE -ne 0) { throw "$method error plot failed" }
        & $python plot_evaluation_overview.py
        if ($LASTEXITCODE -ne 0) { throw "$method overview failed" }
    } finally {
        Pop-Location
    }
}
```

#### 5.3 Resume and options

From the relevant method directory, for example:

```powershell
& $python run_ftnn_50.py --resume
& $python run_ftnn_50.py --help
& $python evaluate_tracks_4_15.py --help
```

`--resume` skips completed runs that pass integrity checks; it does not mean restarting an incomplete run from its last epoch. Preserve the full results directory. Moving existing results can make historical absolute paths fail resume checks. Keep the original results and use a fresh `--output` directory instead of deleting integrity metadata.

|Option |Use |
| --- | --- |
| `--data-root` |Trajectory directory for training and unseen-trajectory evaluation |
| `--lhs-file` |Target LHS file for all five methods |
| `--source-root` |Source-model root for FTNN, RCNN, LC |
| `--low-file` |Low-fidelity Excel for MFDNN |
| `--epochs` |Epochs for the four neural-network methods; not applicable to LC |
| `--results` |Training-results directory for evaluation and plotting |
| `--heldout` |Unseen-trajectory evaluation directory for overview plotting |
| `--output` |Output directory of the current script |

If training uses a custom `--output`, pass that directory as `--results` to later scripts. If unseen-trajectory evaluation uses a custom `--output`, pass it as `--heldout` to the overview plotter. Retain 3000 epochs and the original data for the paper protocol.

### 6. Outputs and paper-result aggregation

All default outputs reside inside the corresponding method directory.

|Output |Contents |
| --- | --- |
| `<results>/experiment_config.json` |Experiment configuration and input audit |
| `<results>/all_run_metrics.csv` |Per-run metrics before aggregation across seeds |
| `<results>/source_<seed>/lr_<rate>/` |Individual FTNN and RCNN runs |
| `<results>/seed_<seed>/lr_<rate>/` |Individual DNN and MFDNN runs |
| `<results>/source_<seed>/` |Individual LC fits |
| `<results>/plots/` |`rmse_mae_comparison.png`, `rmse_mae_summary.csv`, and plotting configuration |
| `heldout_tracks_4_15/` |Unseen-trajectory predictions, metrics, and `evaluation_config.json` |
| `evaluation_overview/` |Overview figure, `all_run_metrics.csv`, and `metrics_summary.csv` across the three evaluation sets |

To prepare Figure 8 and Table 13:

1. For each neural-network method, choose the learning rate with the lowest mean LHS RMSE across ten runs.
2. Apply that same selected rate to every evaluation set for that method; do not select rates separately per set.
3. Pool the predictions from 4° and 15° within each run before computing errors, then report the mean and sample standard deviation across ten runs (`ddof=1`). Do not average the two trajectory RMSEs directly.
4. Aggregate the ten LC fits without a learning-rate search. Obtain the source-dynamics baseline separately.

Table 13 reports FTNN `1e-3`, DNN `1e-4`, RCNN `1e-2`, and MFDNN `5e-3`. These are manuscript results to compare against local reruns, not enforced outcomes. The reported full-fine-tuning FTNN LHS RMSE is `3.25 ± 0.53 m/s`; the abstract's `2.12 ± 0.22 m/s` belongs to the later A3 layer-freezing experiment.

### 7. Validation scope

On November 16, 2026, training and evaluation were completed for all five methods. FTNN and MFDNN were then each rerun for 50 runs of 3,000 epochs in the original-version environment above. Their complete training histories, evaluation metrics (excluding elapsed time), test/LHS predictions, and unseen-trajectory predictions matched the original experiment numerically, with a maximum absolute difference of zero across the values compared.

The LHS RMSE values were `3.256400 ± 0.534217 m/s` for FTNN and `6.928684 ± 3.319786 m/s` for MFDNN. Truncation as used in the paper gives `3.25 ± 0.53` and `6.92 ± 3.31`, respectively. Selected learning rates were `0.001` and `0.005`. The complete DNN, RCNN, and LC runs also agreed at the paper's displayed precision; these three methods were not rerun in the original-version environment during this comparison.

Python 3.12 / TensorFlow 2.21 / NumPy 2.5 previously produced different FTNN and MFDNN values; restoring the original-version environment eliminated those differences. This verification applies to the local CPU setup and supplied code, data, and source weights, and does not guarantee bitwise agreement on arbitrary platforms or dependency versions.

