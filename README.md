# FTNN 迁移学习复现（预训练 + 微调）

[中文](#中文) | [English](#english)

## 中文

本目录用于复现论文的 FTNN 方法：先训练源域预训练模型，再用目标域数据微调，最后用未参与训练的 4°、15° 轨迹评估。其余方法（DNN、RCNN、LC、MFDNN）和源物理基线放在 `appendix/`，作为附属内容。

以下命令中的 `$python` 指向本目录 `.venv` 中的 Python；若尚未定义，先执行 `$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path`。

### 1. 目录内容

```text
FTNN-review/
├── Pretrain-MODEL/     # 源域预训练 PCNN：训练/绘图脚本、训练数据与 10 套源权重
├── FTNN-model/         # FTNN 微调与未见轨迹评估脚本
├── inputdata/          # 所有输入数据（轨迹、LHS、源/目标模型）
├── appendix/           # 其余方法与源物理基线（附属内容）
├── requirements-reproduction.txt
└── README.md
```

- `Pretrain-MODEL/`
  - `train_pcnn_5D.py`：读取 `flightdata/out_747-800_3.xlsx`，训练 10 个初始化种子（1000–1009），结果写入 `out_pcnn_5D/out_<seed>`。
  - `out_pcnn_5D/`：已训练好的 10 套源模型权重（`out_1000`–`out_1009`），FTNN 微调从这里加载。
  - `plot_pcnn_5D.py`、`plot_training_loss.py`：绘图脚本（可选）。
- `FTNN-model/`
  - `run_ftnn_50.py`：10 个源模型 × 5 个学习率 = 50 次微调。
  - `evaluate_tracks_4_15.py`：用未见轨迹评估训练结果。
- `inputdata/`
  - `flightdata/flightdata_split/`：21 个轨迹 Excel 文件（训练/验证/测试划分）。
  - `lhs_target/`：目标域 LHS 文件；`lhs_source/`、`source-model-trim/`、`target-model-trim/` 等为附带数据。

### 2. 如何运行预训练模型

预训练脚本自包含：默认读取 `Pretrain-MODEL/flightdata/out_747-800_3.xlsx`，输出到 `Pretrain-MODEL/out_pcnn_5D/`。

```powershell
Push-Location "Pretrain-MODEL"
& $python train_pcnn_5D.py
Pop-Location
```

- 默认每个种子最多 5000 轮；可用环境变量覆盖，例如 `$env:PCNN_MAX_EPOCHS = "500"`。
- 训练会重新生成 `out_pcnn_5D/out_1000`–`out_1009`；如果只是想用现成的源权重做 FTNN，无需重跑这一步。
- 可选绘图：`& $python plot_pcnn_5D.py`、`& $python plot_training_loss.py`。

### 3. 如何运行 FTNN 模型

需要先具备 `Pretrain-MODEL/out_pcnn_5D/` 中的 10 套源权重（已提供），以及 `inputdata/` 中的轨迹和 LHS 数据。

```powershell
Push-Location "FTNN-model"
& $python run_ftnn_50.py --check-only   # 只检查数据、权重与预测接口，不训练
& $python run_ftnn_50.py                 # 正式训练，输出 results_10sources_5lr
Pop-Location
```

- 默认 3000 轮、全批量 Adam、无早停；50 次微调分别写入 `results_10sources_5lr/source_<seed>/lr_<rate>/`。
- 中断后可用 `--resume` 跳过已完整完成的运行。

### 4. 如何用未见轨迹测试 FTNN 训练结果

先完成第 3 步的训练（生成 `results_10sources_5lr`），再评估未参与训练的 4°、15° 轨迹：

```powershell
Push-Location "FTNN-model"
& $python evaluate_tracks_4_15.py --check-only   # 只校验并推理，不写文件
& $python evaluate_tracks_4_15.py                 # 正式评估，输出 heldout_tracks_4_15
Pop-Location
```

- 评估从 `results_10sources_5lr` 读取已保存的检查点与归一化参数，只做推理、不重新训练。
- 输出 `heldout_tracks_4_15/`：逐点预测、每次运行的 RMSE/MAE、10 次的均值与样本标准差（`ddof=1`）以及图。

### 5. Python 环境

目录内已提供 `.venv`。如需重建环境，在本目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

已验证在 CPU 上运行，不要求 GPU。

---

## English

This directory reproduces the paper's FTNN method: train a source-domain pretrained model, fine-tune it on target-domain data, and evaluate on the unseen 4° and 15° trajectories. The other methods (DNN, RCNN, LC, MFDNN) and the source-physics baseline are kept in `appendix/` as supplementary material.

In the commands below, `$python` refers to the Python in this directory's `.venv`. If it is not set yet, run `$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path` first.

### 1. Contents

```text
FTNN-review/
├── Pretrain-MODEL/     # Source-domain pretrained PCNN: scripts, data, and 10 source weights
├── FTNN-model/         # FTNN fine-tuning and unseen-trajectory evaluation scripts
├── inputdata/          # All input data (trajectories, LHS, source/target models)
├── appendix/           # Other methods and the source-physics baseline (supplementary)
├── requirements-reproduction.txt
└── README.md
```

- `Pretrain-MODEL/`
  - `train_pcnn_5D.py`: reads `flightdata/out_747-800_3.xlsx`, trains 10 initialization seeds (1000–1009), and writes results to `out_pcnn_5D/out_<seed>`.
  - `out_pcnn_5D/`: 10 trained source-model weight sets (`out_1000`–`out_1009`) loaded by FTNN fine-tuning.
  - `plot_pcnn_5D.py`, `plot_training_loss.py`: optional plotting scripts.
- `FTNN-model/`
  - `run_ftnn_50.py`: 10 source models × 5 learning rates = 50 fine-tuning runs.
  - `evaluate_tracks_4_15.py`: evaluates trained results on the unseen trajectories.
- `inputdata/`
  - `flightdata/flightdata_split/`: 21 trajectory Excel files (train/validation/test splits).
  - `lhs_target/`: target-domain LHS files; `lhs_source/`, `source-model-trim/`, `target-model-trim/`, etc. are supplementary.

### 2. Running the pretrained model

The pretraining script is self-contained: it reads `Pretrain-MODEL/flightdata/out_747-800_3.xlsx` and writes to `Pretrain-MODEL/out_pcnn_5D/`.

```powershell
Push-Location "Pretrain-MODEL"
& $python train_pcnn_5D.py
Pop-Location
```

- Each seed uses up to 5000 epochs by default; override via an environment variable, e.g. `$env:PCNN_MAX_EPOCHS = "500"`.
- Training regenerates `out_pcnn_5D/out_1000`–`out_1009`; if you only need the existing source weights for FTNN, skip this step.
- Optional plots: `& $python plot_pcnn_5D.py`, `& $python plot_training_loss.py`.

### 3. Running the FTNN model

This requires the 10 source-weight sets in `Pretrain-MODEL/out_pcnn_5D/` (already provided) and the trajectory/LHS data in `inputdata/`.

```powershell
Push-Location "FTNN-model"
& $python run_ftnn_50.py --check-only   # check data, weights, and prediction interface only
& $python run_ftnn_50.py                 # full training, output results_10sources_5lr
Pop-Location
```

- Defaults: 3000 epochs, full-batch Adam, no early stopping; the 50 fine-tuning runs are written to `results_10sources_5lr/source_<seed>/lr_<rate>/`.
- Use `--resume` after an interruption to skip already-completed runs.

### 4. Testing FTNN results on unseen trajectories

First complete Step 3 (producing `results_10sources_5lr`), then evaluate the unseen 4° and 15° trajectories:

```powershell
Push-Location "FTNN-model"
& $python evaluate_tracks_4_15.py --check-only   # validate and infer without writing files
& $python evaluate_tracks_4_15.py                 # full evaluation, output heldout_tracks_4_15
Pop-Location
```

- Evaluation loads the saved checkpoints and normalization parameters from `results_10sources_5lr`; it only infers and does not retrain.
- Output `heldout_tracks_4_15/`: per-point predictions, per-run RMSE/MAE, and the mean and sample standard deviation over 10 runs (`ddof=1`), plus figures.

### 5. Python environment

A `.venv` is already provided. To rebuild it, run from this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

Runs on CPU; a GPU is not required.
