import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def pyplot_state_grid(output_file_name, input_file_name):
    df = pd.read_csv(output_file_name)
    time = df["time"].values

    # Convert quaternions to euler angles (order: qx, qy, qz, qw for Rotation.from_quat)
    quaternions = df[["q_x", "q_y", "q_z", "q_w"]].values
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler("xyz", degrees=True)
    df["roll"], df["pitch"], df["yaw"] = (
        euler_angles[:, 0],
        euler_angles[:, 1],
        euler_angles[:, 2],
    )

    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    fig.suptitle(input_file_name)

    # Position
    axs[0].plot(time, df["p_x"], label="px", color="red")
    axs[0].plot(time, df["p_y"], label="py", color="green")
    axs[0].plot(time, df["p_z"], label="pz", color="blue")
    axs[0].set_ylabel("position (m)")

    # Velocity
    axs[1].plot(time, df["v_x"], label="vx", color="red")
    axs[1].plot(time, df["v_y"], label="vy", color="green")
    axs[1].plot(time, df["v_z"], label="vz", color="blue")
    axs[1].set_ylabel("velocity (m/s)")

    # Orientation
    axs[2].plot(time, df["roll"], label="Roll", color="blue")
    axs[2].plot(time, df["pitch"], label="Pitch", color="orange")
    axs[2].plot(time, df["yaw"], label="Yaw", color="green")
    axs[2].set_ylabel("orientation (deg)")

    # Accel Bias
    axs[3].plot(time, df["a_b_x"], label="abx", color="red")
    axs[3].plot(time, df["a_b_y"], label="aby", color="green")
    axs[3].plot(time, df["a_b_z"], label="abz", color="blue")
    axs[3].set_ylabel("accel bias")

    # Gyro Bias
    axs[4].plot(time, df["omega_b_x"], label="omega_b_x", color="red")
    axs[4].plot(time, df["omega_b_y"], label="omega_b_y", color="green")
    axs[4].plot(time, df["omega_b_z"], label="omega_b_z", color="blue")
    axs[4].set_ylabel("gyro bias")
    axs[4].set_xlabel("time (s)")

    for ax in axs:
        ax.legend(loc="upper right")
        ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def pyplot_input_and_state_grid(output_file_name, output_columns, input_file_name):
    df = pd.read_csv(output_file_name)
    time = df["time"].values

    # Pre-calculate Euler angles
    quats_df = df[["q_x", "q_y", "q_z", "q_w"]].copy()
    nan_mask = quats_df.isna().any(axis=1)

    # Fill NaNs temporarily with an identity quaternion [0, 0, 0, 1] to prevent SciPy crash
    quats_df["q_x"] = quats_df["q_x"].fillna(0.0)
    quats_df["q_y"] = quats_df["q_y"].fillna(0.0)
    quats_df["q_z"] = quats_df["q_z"].fillna(0.0)
    quats_df["q_w"] = quats_df["q_w"].fillna(1.0)

    # Pre-calculate Euler angles
    rotations = Rotation.from_quat(quats_df.values)
    euler = rotations.as_euler("xyz", degrees=True)

    df["roll"], df["pitch"], df["yaw"] = euler[:, 0], euler[:, 1], euler[:, 2]

    # Set Euler angles to NaN during the calibration phase so they don't plot falsely
    df.loc[nan_mask, ["roll", "pitch", "yaw"]] = np.nan

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(input_file_name)

    # Row 0: Accelerometer, Gyro
    axs[0, 0].plot(time, df["a_tilde_x"], label="ax", color="red")
    axs[0, 0].plot(time, df["a_tilde_y"], label="ay", color="green")
    axs[0, 0].plot(time, df["a_tilde_z"], label="az", color="blue")
    axs[0, 0].set_ylabel("accel (m/s^2)")
    axs[0, 0].set_title("Accelerometer")

    axs[0, 1].plot(time, df["omega_tilde_x"], label="gx", color="red")
    axs[0, 1].plot(time, df["omega_tilde_y"], label="gy", color="green")
    axs[0, 1].plot(time, df["omega_tilde_z"], label="gz", color="blue")
    axs[0, 1].set_ylabel("gyro (rad/s)")
    axs[0, 1].set_title("Gyroscope")

    # Row 1: Position, Orientation
    axs[2, 0].plot(time, df["p_x"], label="px", color="red")
    axs[2, 0].plot(time, df["p_y"], label="py", color="green")
    axs[2, 0].plot(time, df["p_z"], label="pz", color="blue")
    axs[2, 0].set_ylabel("pos (m)")

    axs[1, 1].plot(time, df["roll"], label="Roll")
    axs[1, 1].plot(time, df["pitch"], label="Pitch")
    axs[1, 1].plot(time, df["yaw"], label="Yaw")
    axs[1, 1].set_ylabel("orientation (deg)")

    # Row 2: Velocity, empty space
    axs[1, 0].plot(time, df["v_x"], label="vx", color="red")
    axs[1, 0].plot(time, df["v_y"], label="vy", color="green")
    axs[1, 0].plot(time, df["v_z"], label="vz", color="blue")
    axs[1, 0].set_ylabel("vel (m/s)")

    axs[2, 1].axis("off")  # Empty space

    # Row 3: Accel Bias, Gyro Bias
    axs[3, 0].plot(time, df["a_b_x"], label="abx")
    axs[3, 0].plot(time, df["a_b_y"], label="aby")
    axs[3, 0].plot(time, df["a_b_z"], label="abz")
    axs[3, 0].set_ylabel("accel bias")
    axs[3, 0].set_xlabel("time (s)")

    axs[3, 1].plot(time, df["omega_b_x"], label="omega_b_x")
    axs[3, 1].plot(time, df["omega_b_y"], label="omega_b_y")
    axs[3, 1].plot(time, df["omega_b_z"], label="omega_b_z")
    axs[3, 1].set_ylabel("gyro bias")

    for ax in axs.flat:
        if ax.has_data():
            ax.legend(loc="best", fontsize="small")
            ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
