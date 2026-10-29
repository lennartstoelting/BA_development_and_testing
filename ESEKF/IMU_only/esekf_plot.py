import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from esekf_definitions import *


def pyplot_input_and_state_grid(output_file_name, input_file_name):
    p_sym = get_tex_vec("IMU position")
    v_sym = get_tex_vec("IMU velocity")
    q_sym = get_tex_vec("IMU orientation")
    a_b_sym = get_tex_vec("IMU accel bias")
    omega_b_sym = get_tex_vec("IMU gyro bias")

    df = pd.read_csv(output_file_name)
    time = df["time"].values

    # Pre-calculate Euler angles
    quats_df = df[[q_sym[1], q_sym[2], q_sym[3], q_sym[0]]].copy()
    nan_mask = quats_df.isna().any(axis=1)

    # Fill NaNs temporarily with an identity quaternion [0, 0, 0, 1] to prevent SciPy crash
    quats_df[q_sym[1]] = quats_df[q_sym[1]].fillna(0.0)
    quats_df[q_sym[2]] = quats_df[q_sym[2]].fillna(0.0)
    quats_df[q_sym[3]] = quats_df[q_sym[3]].fillna(0.0)
    quats_df[q_sym[0]] = quats_df[q_sym[0]].fillna(1.0)

    # Pre-calculate Euler angles
    rotations = Rotation.from_quat(quats_df.values)
    euler = rotations.as_euler("xyz", degrees=True)

    df["roll"], df["pitch"], df["yaw"] = euler[:, 0], euler[:, 1], euler[:, 2]

    # Set Euler angles to NaN during the calibration phase so they don't plot falsely
    df.loc[nan_mask, ["roll", "pitch", "yaw"]] = np.nan

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(input_file_name)

    # Row 0: Accelerometer and Gyro
    axs[0, 0].plot(time, df["a_tilde_x"], label=r"$\tilde{a}_x$", color="red")
    axs[0, 0].plot(time, df["a_tilde_y"], label=r"$\tilde{a}_y$", color="green")
    axs[0, 0].plot(time, df["a_tilde_z"], label=r"$\tilde{a}_z$", color="blue")
    axs[0, 0].set_ylabel("Accelerometer (g)")

    axs[0, 1].plot(time, df["omega_tilde_x"], label=r"$\tilde{\omega}_x$", color="red")
    axs[0, 1].plot(
        time, df["omega_tilde_y"], label=r"$\tilde{\omega}_y$", color="green"
    )
    axs[0, 1].plot(time, df["omega_tilde_z"], label=r"$\tilde{\omega}_z$", color="blue")
    axs[0, 1].set_ylabel("Gyroscope (deg/s)")

    # Row 1: Position and Orientation
    axs[2, 0].plot(time, df[p_sym[0]], label=r"$p_x$", color="red")
    axs[2, 0].plot(time, df[p_sym[1]], label=r"$p_y$", color="green")
    axs[2, 0].plot(time, df[p_sym[2]], label=r"$p_z$", color="blue")
    axs[2, 0].set_ylabel("Position (m)")

    axs[1, 1].plot(time, df["roll"], label="Roll")
    axs[1, 1].plot(time, df["pitch"], label="Pitch")
    axs[1, 1].plot(time, df["yaw"], label="Yaw")
    axs[1, 1].set_ylabel("Orientation (deg)")

    # Row 2: Velocity and empty space
    axs[1, 0].plot(time, df[v_sym[0]], label=r"$v_x$", color="red")
    axs[1, 0].plot(time, df[v_sym[1]], label=r"$v_y$", color="green")
    axs[1, 0].plot(time, df[v_sym[2]], label=r"$v_z$", color="blue")
    axs[1, 0].set_ylabel("Velocity (m/s)")

    axs[2, 1].axis("off")  # Empty space

    # Row 3: Accel Bias and Gyro Bias
    axs[3, 0].plot(time, df[a_b_sym[0]], label=r"$a_{b,x}$")
    axs[3, 0].plot(time, df[a_b_sym[1]], label=r"$a_{b,y}$")
    axs[3, 0].plot(time, df[a_b_sym[2]], label=r"$a_{b,z}$")
    axs[3, 0].set_ylabel("accel bias")
    axs[3, 0].set_xlabel("time (s)")

    axs[3, 1].plot(time, df[omega_b_sym[0]], label=r"$\omega_{b,x}$")
    axs[3, 1].plot(time, df[omega_b_sym[1]], label=r"$\omega_{b,y}$")
    axs[3, 1].plot(time, df[omega_b_sym[2]], label=r"$\omega_{b,z}$")
    axs[3, 1].set_ylabel("gyro bias")
    axs[3, 0].set_xlabel("time (s)")

    for ax in axs.flat:
        if ax.has_data():
            ax.legend(loc="best", fontsize="small")
            ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
