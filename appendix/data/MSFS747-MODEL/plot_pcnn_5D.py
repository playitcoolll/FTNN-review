"""交互选择 m、dt、tem，在 Mach-Height 平面绘制五维模型预测结果。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from train_pcnn_5D import (
    CELSIUS_TO_KELVIN,
    DT_SCALE,
    PretrainedNN,
    build_mach_boundary_constraint_points,
    HEIGHT_SCALE,
    MASS_SCALE,
    STANDARD_TEMPERATURE_K,
    tf,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "out_pcnn_5D"
PLOT_DIR = MODEL_DIR / "plots"
LAYERS = [5, 64, 64, 64, 1]
DEFAULT_GRID_POINTS = 150
MACH_PLOT_MIN = 0.15
MACH_PLOT_MAX = 0.85
HEIGHT_PLOT_MIN = 400.0
HEIGHT_PLOT_MAX = 12500.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_results(model_dir: Path):
    """读取实际训练数据和 TensorFlow 检查点。"""
    predictions_path = model_dir / "training_predictions.csv"
    checkpoint_path = model_dir / "pcnn_model.ckpt"
    required = [predictions_path, checkpoint_path.with_suffix(".ckpt.index")]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"缺少 {names}，请先运行 train_pcnn_5D.py 完成训练"
        )

    predictions = pd.read_csv(predictions_path)
    required_columns = {
        "mach", "height_m", "m", "dt", "tem", "ps_true"
    }
    absent = sorted(required_columns.difference(predictions.columns))
    if absent:
        raise ValueError(f"training_predictions.csv 缺少列：{absent}")
    return predictions, checkpoint_path


def build_scaled_input(
    mach: np.ndarray,
    height: np.ndarray,
    mass: np.ndarray,
    dt: np.ndarray,
    temperature_celsius: np.ndarray,
) -> np.ndarray:
    """执行与训练脚本完全一致的五维输入缩放。"""
    return np.column_stack(
        [
            mach,
            height / HEIGHT_SCALE,
            mass / MASS_SCALE,
            dt / DT_SCALE,
            (temperature_celsius + CELSIUS_TO_KELVIN)
            / STANDARD_TEMPERATURE_K,
        ]
    ).astype(np.float32)


def restore_model(predictions: pd.DataFrame, checkpoint_path: Path):
    """按照训练时的输入范围重建网络并恢复权重。"""
    x_train = build_scaled_input(
        predictions["mach"].to_numpy(dtype=np.float32),
        predictions["height_m"].to_numpy(dtype=np.float32),
        predictions["m"].to_numpy(dtype=np.float32),
        predictions["dt"].to_numpy(dtype=np.float32),
        predictions["tem"].to_numpy(dtype=np.float32),
    )
    y_train = predictions["ps_true"].to_numpy(dtype=np.float32).reshape(-1, 1)
    lb = x_train.min(axis=0)
    ub = x_train.max(axis=0)

    x_mach_left = build_mach_boundary_constraint_points(x_train)
    model = PretrainedNN(x_train, y_train, x_mach_left, LAYERS, lb, ub)
    # 只恢复网络权重和偏置；作图不需要 Adam 优化器状态。
    saver = tf.train.Saver(var_list=model.weights + model.biases)
    saver.restore(model.sess, str(checkpoint_path))
    return model, lb, ub


def prompt_value(name: str, unit: str, available: np.ndarray) -> float:
    """显示训练数据中的可用取值，并循环读取一个数值。"""
    values = ", ".join(f"{value:g}" for value in np.sort(np.unique(available)))
    print(f"训练数据中的 {name}：{values}")
    while True:
        raw = input(f"请输入 {name}{unit}：").strip()
        try:
            return float(raw)
        except ValueError:
            print("输入无效，请输入一个数值。")


def evaluate_plane(
    model,
    predictions: pd.DataFrame,
    mass: float,
    dt: float,
    temperature_celsius: float,
    grid_points: int,
):
    """固定 m、dt、tem，在 Mach-Height 规则网格上预测 Ps。"""
    mach_values = np.linspace(MACH_PLOT_MIN, MACH_PLOT_MAX, grid_points)
    height_values = np.linspace(
        HEIGHT_PLOT_MIN,
        HEIGHT_PLOT_MAX,
        grid_points,
    )
    mach_grid, height_grid = np.meshgrid(mach_values, height_values)
    point_count = mach_grid.size
    grid_input = build_scaled_input(
        mach_grid.ravel(),
        height_grid.ravel(),
        np.full(point_count, mass, dtype=np.float32),
        np.full(point_count, dt, dtype=np.float32),
        np.full(point_count, temperature_celsius, dtype=np.float32),
    )
    ps_grid = model.predict(grid_input).reshape(mach_grid.shape)
    return mach_grid, height_grid, ps_grid, grid_input


def warn_if_extrapolating(grid_input: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """提示固定工况是否超出模型训练输入范围。"""
    fixed_names = np.asarray(["m", "dt", "tem"])
    fixed_columns = grid_input[0, 2:5]
    outside = (fixed_columns < lb[2:5]) | (fixed_columns > ub[2:5])
    if np.any(outside):
        names = ", ".join(fixed_names[outside])
        print(f"警告：{names} 超出训练范围，当前平面属于外推预测。")


def number_tag(value: float) -> str:
    """将数值转换为适合文件名的短字符串。"""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_plane(
    mach_grid: np.ndarray,
    height_grid: np.ndarray,
    ps_grid: np.ndarray,
    predictions: pd.DataFrame,
    mass: float,
    dt: float,
    temperature_celsius: float,
    output_dir: Path,
) -> Path:
    """绘制预测等值色图。"""
    value_min = float(np.nanmin(ps_grid))
    value_max = float(np.nanmax(ps_grid))
    if np.isclose(value_min, value_max):
        value_min -= 1e-6
        value_max += 1e-6
    norm = Normalize(vmin=value_min, vmax=value_max)
    levels = np.linspace(value_min, value_max, 60)

    fig, ax = plt.subplots(figsize=(9, 7))
    filled = ax.contourf(
        mach_grid,
        height_grid,
        ps_grid,
        levels=levels,
        cmap="viridis",
        norm=norm,
        extend="both",
    )
    contour_lines = ax.contour(
        mach_grid,
        height_grid,
        ps_grid,
        levels=[-10, -5, 0, 5, 10],
        colors="black",
        linewidths=0.5,
        alpha=0.45,
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.0f")

    fitting_mask = (
        np.isclose(predictions["m"], mass)
        & np.isclose(predictions["dt"], dt)
        & np.isclose(predictions["tem"], temperature_celsius)
    )
    fitting_points = predictions.loc[fitting_mask]
    if not fitting_points.empty:
        ax.scatter(
            fitting_points["mach"],
            fitting_points["height_m"],
            c=fitting_points["ps_true"],
            cmap="viridis",
            norm=norm,
            s=34,
            edgecolors="white",
            linewidths=0.7,
            label=f"Fitting data (n={len(fitting_points)})",
            zorder=3,
        )
        ax.legend(loc="best", framealpha=0.85)
    else:
        print("提示：当前 m、dt、tem 组合没有对应的拟合散点。")

    ax.set_title(
        "五维模型的 Mach-Height 预测平面\n"
        f"m={mass:g}, dt={dt:g}, tem={temperature_celsius:g} °C"
    )
    ax.set_xlabel("Mach")
    ax.set_ylabel("Height (m)")
    ax.set_xlim(MACH_PLOT_MIN, MACH_PLOT_MAX)
    ax.set_ylim(HEIGHT_PLOT_MIN, HEIGHT_PLOT_MAX)
    fig.colorbar(filled, ax=ax, label="Ps（预测值）")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pcnn_5D_m_{number_tag(mass)}_dt_{number_tag(dt)}_"
        f"tem_{number_tag(temperature_celsius)}.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return output_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="固定 m、dt、tem，绘制五维模型的 Mach-Height 预测平面。"
    )
    parser.add_argument("--m", dest="mass", type=float, help="质量 m（原始单位）")
    parser.add_argument("--dt", type=float, help="dt（原始单位）")
    parser.add_argument("--tem", type=float, help="摄氏温度")
    parser.add_argument(
        "--grid-points",
        type=int,
        default=DEFAULT_GRID_POINTS,
        help=f"每个坐标方向的网格数，默认 {DEFAULT_GRID_POINTS}",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="只保存图片，不弹出绘图窗口"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.grid_points < 2:
        raise ValueError("--grid-points 必须大于等于 2")

    predictions, checkpoint_path = load_results(MODEL_DIR)

    mass = args.mass
    if mass is None:
        mass = prompt_value("m", "（原始单位）", predictions["m"].to_numpy())
    dt = args.dt
    if dt is None:
        dt = prompt_value("dt", "（原始单位）", predictions["dt"].to_numpy())
    temperature_celsius = args.tem
    if temperature_celsius is None:
        temperature_celsius = prompt_value(
            "tem", "（°C）", predictions["tem"].to_numpy()
        )

    tf.reset_default_graph()
    model, lb, ub = restore_model(predictions, checkpoint_path)
    try:
        mach_grid, height_grid, ps_grid, grid_input = evaluate_plane(
            model,
            predictions,
            mass,
            dt,
            temperature_celsius,
            args.grid_points,
        )
        warn_if_extrapolating(grid_input, lb, ub)
        output_path = plot_plane(
            mach_grid,
            height_grid,
            ps_grid,
            predictions,
            mass,
            dt,
            temperature_celsius,
            PLOT_DIR,
        )
        print(f"图片已保存：{output_path}")
        if not args.no_show:
            plt.show()
    finally:
        model.sess.close()


if __name__ == "__main__":
    main()
