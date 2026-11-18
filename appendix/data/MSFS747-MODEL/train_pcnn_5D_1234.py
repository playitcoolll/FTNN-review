import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from sklearn.model_selection import train_test_split

np.random.seed(1234)
tf.disable_v2_behavior()
tf.set_random_seed(1234)


HEIGHT_SCALE = 10000.0
MASS_SCALE = 100000.0
DT_SCALE = 100.0
CELSIUS_TO_KELVIN = 273.15
STANDARD_TEMPERATURE_K = 273.15

LEARNING_RATE = 1e-3
MAX_EPOCHS = int(os.getenv("PCNN_MAX_EPOCHS", "5000"))
PRINT_EVERY = 500
EARLY_STOP_LOSS = 1e-2
LAMBDA_MACH_LEFT = float(os.getenv("PCNN_LAMBDA_MACH_LEFT", "0.01"))
LAMBDA_MACH_LEFT_SECOND = float(os.getenv("PCNN_LAMBDA_MACH_LEFT_SECOND", "0.01"))
MACH_LEFT_SAMPLES = int(os.getenv("PCNN_MACH_LEFT_SAMPLES", "20"))
MACH_DERIVATIVE_MIN = float(os.getenv("PCNN_MACH_DERIVATIVE_MIN", "1e-10"))
MACH_SECOND_DERIVATIVE_MAX = float(os.getenv("PCNN_MACH_SECOND_DERIVATIVE_MAX", "0.0"))
HP_VALUES = (457, 3048, 6096, 9144, 12192)
MACH_VALUES = (0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MASS_VALUES = tuple(range(240000, 300001, 10000))
TEM_VALUES = tuple(range(-20, 21, 10))
N1_VALUES = (70, 80, 90)


def build_mach_boundary_constraint_points(x_data):
    if MACH_LEFT_SAMPLES < 1:
        raise ValueError("Mach 左边界采样数必须大于等于 1")

    x_data = np.asarray(x_data, dtype=np.float32)
    global_mach_min = float(np.min(x_data[:, 0]))
    condition_values = np.unique(x_data[:, 1:5], axis=0)
    left_points = []

    for condition in condition_values:
        mask = np.all(np.isclose(x_data[:, 1:5], condition), axis=1)
        group_mach = x_data[mask, 0]
        left_boundary = float(np.min(group_mach))

        if np.isclose(left_boundary, global_mach_min):
            continue
        left_mach = np.linspace(
            global_mach_min,
            left_boundary,
            MACH_LEFT_SAMPLES,
            endpoint=False,
            dtype=np.float32,
        )
        left_points.extend(
            np.concatenate(([mach_value], condition)) for mach_value in left_mach
        )

    if not left_points:
        raise ValueError("当前数据没有可用的 Mach 左边界外侧约束区间")
    return np.asarray(left_points, dtype=np.float32)

class PretrainedNN:
    def __init__(self, X_u, u, X_mach_left, layers, lb, ub):

        self.X_u = X_u
        self.u = u
        self.X_mach_left = X_mach_left
        self.lb = lb
        self.ub = ub

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False
        ))

        self.X_u_tf = tf.placeholder(tf.float32, shape=[None, 5])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.X_mach_left_tf = tf.placeholder(tf.float32, shape=[None, 5])

        self.u_pred = self.net_u(self.X_u_tf)
        self.mach_left_pred = self.net_u(self.X_mach_left_tf)
        self.du_dmach_left = tf.gradients(
            self.mach_left_pred, self.X_mach_left_tf
        )[0][:, 0:1]
        self.d2u_dmach2_left = tf.gradients(
            self.du_dmach_left, self.X_mach_left_tf
        )[0][:, 0:1]

        self.loss_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_mach_left = tf.reduce_mean(
            tf.square(tf.nn.relu(MACH_DERIVATIVE_MIN - self.du_dmach_left))
        )
        self.loss_mach_left_second = tf.reduce_mean(
            tf.square(
                tf.nn.relu(self.d2u_dmach2_left - MACH_SECOND_DERIVATIVE_MAX)
            )
        )
        self.violation_mach_left = tf.reduce_mean(
            tf.cast(self.du_dmach_left < MACH_DERIVATIVE_MIN, tf.float32)
        )
        self.violation_mach_left_second = tf.reduce_mean(
            tf.cast(
                self.d2u_dmach2_left > MACH_SECOND_DERIVATIVE_MAX,
                tf.float32,
            )
        )
        self.loss = (
            self.loss_data
            + LAMBDA_MACH_LEFT * self.loss_mach_left
            + LAMBDA_MACH_LEFT_SECOND * self.loss_mach_left_second
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for index in range(len(layers) - 1):
            weights.append(self.xavier_init([layers[index], layers[index + 1]]))
            biases.append(tf.Variable(
                tf.zeros([1, layers[index + 1]], dtype=tf.float32)
            ))
        return weights, biases

    def xavier_init(self, size):
        stddev = np.sqrt(2 / (size[0] + size[1]))
        return tf.Variable(tf.truncated_normal(size, stddev=stddev))

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for index in range(len(self.weights) - 1):
            H = tf.tanh(tf.add(
                tf.matmul(H, self.weights[index]), self.biases[index]
            ))
        return tf.add(tf.matmul(H, self.weights[-1]), self.biases[-1])

    def net_u(self, X):
        return self.neural_net(X)

    def feed_dict(self):
        return {
            self.X_u_tf: self.X_u,
            self.u_tf: self.u,
            self.X_mach_left_tf: self.X_mach_left,
        }

    def train(self, X_validation, u_validation):
        tf_dict = self.feed_dict()
        validation_dict = {
            self.X_u_tf: np.asarray(X_validation, dtype=np.float32),
            self.u_tf: np.asarray(u_validation, dtype=np.float32).reshape(-1, 1),
        }
        history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            # Evaluate both splits after the same update. Validation is never
            # passed to the optimizer; its loss is data MSE without constraints.
            self.sess.run(self.train_op, feed_dict=tf_dict)
            (
                loss,
                loss_data,
                loss_mach_left,
                loss_mach_left_second,
                violation_mach_left,
                violation_mach_left_second,
            ) = self.sess.run(
                [
                    self.loss,
                    self.loss_data,
                    self.loss_mach_left,
                    self.loss_mach_left_second,
                    self.violation_mach_left,
                    self.violation_mach_left_second,
                ],
                feed_dict=tf_dict,
            )
            val_loss = self.sess.run(self.loss_data, feed_dict=validation_dict)
            weighted_mach_left = LAMBDA_MACH_LEFT * loss_mach_left
            weighted_mach_left_second = (
                LAMBDA_MACH_LEFT_SECOND * loss_mach_left_second
            )
            history.append(
                [
                    epoch,
                    loss,
                    loss_data,
                    weighted_mach_left,
                    weighted_mach_left_second,
                    violation_mach_left,
                    violation_mach_left_second,
                    val_loss,
                ]
            )
            if epoch == 1 or epoch % PRINT_EVERY == 0:
                print(
                    "Epoch %5d: total=%.6e, data=%.6e, val_mse=%.6e, "
                    "mach_left=%.3e, mach_left_second=%.3e, "
                    "violations=(%.1f%%, %.1f%%)"
                    % (
                        epoch,
                        loss,
                        loss_data,
                        val_loss,
                        weighted_mach_left,
                        weighted_mach_left_second,
                        100.0 * violation_mach_left,
                        100.0 * violation_mach_left_second,
                    )
                )
            if loss < EARLY_STOP_LOSS:
                print(
                    "Early stopping at epoch %d: loss=%.6e < %.6e"
                    % (epoch, loss, EARLY_STOP_LOSS)
                )
                break
        return np.asarray(history)

    def predict(self, X_star):
        return self.sess.run(self.u_pred, {self.X_u_tf: X_star})

    def evaluate_mach_boundary_derivatives(self):
        return self.sess.run(
            [self.du_dmach_left, self.d2u_dmach2_left],
            {self.X_mach_left_tf: self.X_mach_left},
        )


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "flightdata" / "out_747-800_3.xlsx"
    output_dir = script_dir / "out_pcnn_5D"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(data_file, engine="openpyxl").round(2)
    required_columns = {"success", "h", "mach", "m", "tem", "dt", "hdot"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError("理论数据缺少列：%s" % missing_columns)
    selected = (
        (data["success"].astype(str).str.upper() == "TRUE")
        & data["h"].isin(HP_VALUES)
        & data["mach"].isin(MACH_VALUES)
        & data["m"].isin(MASS_VALUES)
        & data["tem"].isin(TEM_VALUES)
        & data["dt"].isin(N1_VALUES)
    )
    filtered = data[selected]
    if filtered.empty:
        raise ValueError("没有找到符合指定 Hp、Mach、Mass、ΔTem、N1 组合的数据")
    
    mach = filtered["mach"].to_numpy(dtype=np.float32)
    height = filtered["h"].to_numpy(dtype=np.float32)
    mass = filtered["m"].to_numpy(dtype=np.float32)
    dt = filtered["dt"].to_numpy(dtype=np.float32)
    tem = filtered["tem"].to_numpy(dtype=np.float32)
    ps = -filtered["hdot"].to_numpy(dtype=np.float32)
    valid = (
        np.isfinite(mach)
        & np.isfinite(height)
        & np.isfinite(mass)
        & np.isfinite(dt)
        & np.isfinite(tem)
        & np.isfinite(ps)
    )
    mach, height, mass, dt, tem, ps = (
        values[valid] for values in (mach, height, mass, dt, tem, ps)
    )

    all_indices = np.arange(len(ps))
    train_indices, remaining_indices = train_test_split(
        all_indices, test_size=0.30, random_state=1234, shuffle=True
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices, test_size=1.0 / 3.0, random_state=1234, shuffle=True
    )
    mach_all, height_all, mass_all, dt_all, tem_all, ps_all = (
        mach.copy(), height.copy(), mass.copy(), dt.copy(), tem.copy(), ps.copy()
    )
    mach, height, mass, dt, tem, ps = (
        values[train_indices]
        for values in (mach_all, height_all, mass_all, dt_all, tem_all, ps_all)
    )

    temperature_kelvin = tem + CELSIUS_TO_KELVIN
    X_u_train = np.column_stack(
        [
            mach,
            height / HEIGHT_SCALE,
            mass / MASS_SCALE,
            dt / DT_SCALE,
            temperature_kelvin / STANDARD_TEMPERATURE_K,
        ]
    ).astype(np.float32)
    u_train = ps.reshape(-1, 1).astype(np.float32)
    X_u_validation = np.column_stack(
        [
            mach_all[validation_indices],
            height_all[validation_indices] / HEIGHT_SCALE,
            mass_all[validation_indices] / MASS_SCALE,
            dt_all[validation_indices] / DT_SCALE,
            (tem_all[validation_indices] + CELSIUS_TO_KELVIN)
            / STANDARD_TEMPERATURE_K,
        ]
    ).astype(np.float32)
    ps_validation = ps_all[validation_indices]
    X_u_test = np.column_stack(
        [
            mach_all[test_indices],
            height_all[test_indices] / HEIGHT_SCALE,
            mass_all[test_indices] / MASS_SCALE,
            dt_all[test_indices] / DT_SCALE,
            (tem_all[test_indices] + CELSIUS_TO_KELVIN)
            / STANDARD_TEMPERATURE_K,
        ]
    ).astype(np.float32)
    ps_test = ps_all[test_indices]

    lb = X_u_train.min(axis=0)
    ub = X_u_train.max(axis=0)

    if np.any(ub == lb):
        names = np.asarray(["mach", "height", "m", "dt", "tem"])[ub == lb]
        raise ValueError("以下输入维度只有一个取值：%s" % names.tolist())

    X_mach_left = build_mach_boundary_constraint_points(X_u_train)

    layers = [5, 64, 64, 64, 1]
    model = PretrainedNN(
        X_u_train,
        u_train,
        X_mach_left,
        layers,
        lb,
        ub,
    )

    print("m/dt/tem 使用全部有效取值，不再固定")
    print(
        "输入缩放：Height/%.0f，m/%.0f，dt/%.0f，"
        "Temperature=(tem+%.2f)/%.2f"
        % (
            HEIGHT_SCALE,
            MASS_SCALE,
            DT_SCALE,
            CELSIUS_TO_KELVIN,
            STANDARD_TEMPERATURE_K,
        )
    )
    print(
        "数据划分：训练=%d，验证=%d，测试=%d"
        % (len(X_u_train), len(X_u_validation), len(X_u_test))
    )
    print("Mach 左边界外侧约束点：%d" % len(X_mach_left))
    print(
        "损失函数：data MSE + 左边界外侧 dPs/dMach >= %.1e + d2Ps/dMach2 <= %.1e"
        % (MACH_DERIVATIVE_MIN, MACH_SECOND_DERIVATIVE_MAX)
    )
    print(
        "约束权重：mach_left=%g, mach_left_second=%g"
        % (LAMBDA_MACH_LEFT, LAMBDA_MACH_LEFT_SECOND)
    )


    start_time = time.time()
    history = model.train(X_u_validation, ps_validation)
    elapsed = time.time() - start_time

    y_pred = model.predict(X_u_train).flatten()
    y_validation_pred = model.predict(X_u_validation).flatten()
    y_test_pred = model.predict(X_u_test).flatten()
    mach_derivatives = model.evaluate_mach_boundary_derivatives()
    mach_first_derivatives = mach_derivatives[0].reshape(-1)
    mach_second_derivatives = mach_derivatives[1].reshape(-1)
    relative_l2 = np.linalg.norm(ps - y_pred, 2) / np.linalg.norm(ps, 2)
    rmse = np.sqrt(np.mean((ps - y_pred) ** 2))
    mae = np.mean(np.abs(ps - y_pred))
    r2 = 1.0 - np.sum((ps - y_pred) ** 2) / np.sum((ps - np.mean(ps)) ** 2)
    validation_rmse = np.sqrt(
        np.mean((ps_validation - y_validation_pred) ** 2)
    )
    validation_mae = np.mean(np.abs(ps_validation - y_validation_pred))
    validation_r2 = 1.0 - np.sum(
        (ps_validation - y_validation_pred) ** 2
    ) / np.sum((ps_validation - np.mean(ps_validation)) ** 2)
    test_rmse = np.sqrt(np.mean((ps_test - y_test_pred) ** 2))
    test_mae = np.mean(np.abs(ps_test - y_test_pred))
    test_r2 = 1.0 - np.sum((ps_test - y_test_pred) ** 2) / np.sum(
        (ps_test - np.mean(ps_test)) ** 2
    )

    pd.DataFrame(
        history,
        columns=[
            "epoch",
            "loss",
            "loss_data",
            "loss_mach_left",
            "loss_mach_left_second",
            "mach_left_violation_rate",
            "mach_left_second_violation_rate",
            "val_loss",
        ],
    ).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "mach": mach,
            "height_m": height,
            "m": mass,
            "dt": dt,
            "tem": tem,
            "ps_true": ps,
            "ps_pred": y_pred,
            "residual": ps - y_pred,
        }
    ).to_csv(output_dir / "training_predictions.csv", index=False)
    pd.DataFrame(
        {
            "mach": mach_all[validation_indices],
            "height_m": height_all[validation_indices],
            "m": mass_all[validation_indices],
            "dt": dt_all[validation_indices],
            "tem": tem_all[validation_indices],
            "ps_true": ps_validation,
            "ps_pred": y_validation_pred,
            "residual": ps_validation - y_validation_pred,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "mach": mach_all[test_indices],
            "height_m": height_all[test_indices],
            "m": mass_all[test_indices],
            "dt": dt_all[test_indices],
            "tem": tem_all[test_indices],
            "ps_true": ps_test,
            "ps_pred": y_test_pred,
            "residual": ps_test - y_test_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"split": "train", "rmse": rmse, "mae": mae, "r2": r2, "sample_count": len(ps)},
            {"split": "validation", "rmse": validation_rmse, "mae": validation_mae, "r2": validation_r2, "sample_count": len(ps_validation)},
            {"split": "test", "rmse": test_rmse, "mae": test_mae, "r2": test_r2, "sample_count": len(ps_test)},
        ]
    ).to_csv(output_dir / "evaluation_metrics.csv", index=False)
    pd.DataFrame(
        {
            "mach": X_mach_left[:, 0],
            "height_m": X_mach_left[:, 1] * HEIGHT_SCALE,
            "m": X_mach_left[:, 2] * MASS_SCALE,
            "dt": X_mach_left[:, 3] * DT_SCALE,
            "tem": (
                X_mach_left[:, 4] * STANDARD_TEMPERATURE_K
                - CELSIUS_TO_KELVIN
            ),
            "side": "left_outside",
            "dps_dmach": mach_first_derivatives,
            "d2ps_dmach2": mach_second_derivatives,
            "violated": mach_first_derivatives < MACH_DERIVATIVE_MIN,
            "second_violated": (
                mach_second_derivatives > MACH_SECOND_DERIVATIVE_MAX
            ),
        }
    ).to_csv(output_dir / "mach_boundary_derivatives.csv", index=False)
    saver = tf.train.Saver()
    saver.save(
        model.sess,
        str(output_dir / "pcnn_model.ckpt"),
        write_meta_graph=False,
    )

    print("\n========== 五维 PCNN 训练结果 ==========")
    print("RMSE:               %.6f" % rmse)
    print("R2:                 %.6f" % r2)
    print("Relative L2:        %.6e" % relative_l2)
    print("Validation RMSE:    %.6f" % validation_rmse)
    print("Validation MAE:     %.6f" % validation_mae)
    print("Validation R2:      %.6f" % validation_r2)
    print("Test RMSE:          %.6f" % test_rmse)
    print("Test MAE:           %.6f" % test_mae)
    print("Test R2:            %.6f" % test_r2)
    print("Training time:      %.2f s" % elapsed)
    print("结果目录:           %s" % output_dir)

    from plot_training_loss import main as plot_training_loss
    plot_training_loss()
