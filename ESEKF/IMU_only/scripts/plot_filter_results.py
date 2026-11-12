import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import sys

# go to parent directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import *


def pyplot_input_and_state_grid(output_file_name, input_file_name):
    q_sym = get_def("IMU orientation")["dims"]

    df = pd.read_csv(output_file_name)
    time = df["time"].values

    quats_df = df[q_sym].copy()

    # Pre-calculate Euler angles
    rotations = Rotation.from_quat(quats_df.values, scalar_first=True)
    euler = rotations.as_euler("yxz", degrees=True)

    df["roll"], df["pitch"], df["yaw"] = euler[:, 0], euler[:, 1], euler[:, 2]

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(input_file_name)

    plots = [
        {"name": "IMU accelerometer", "row": 0, "col": 0},
        {"name": "IMU gyroscope", "row": 0, "col": 1},
        {"name": "IMU velocity", "row": 1, "col": 0},
        {"name": "IMU orientation", "row": 1, "col": 1},
        {"name": "IMU position", "row": 2, "col": 0},
        {"name": "empty plot"},
        {"name": "IMU accel bias", "row": 3, "col": 0},
        {"name": "IMU gyro bias", "row": 3, "col": 1},
    ]

    for plot in plots:
        if plot["name"] == "IMU orientation" or plot["name"] == "empty plot":
            continue
        for index, curve in enumerate(["red", "green", "blue"]):
            latex_string = get_def(plot["name"])["dims"][index]
            axs[plot["row"], plot["col"]].plot(
                time, df[latex_string], label="$" + latex_string + "$", color=curve
            )
        axs[plot["row"], plot["col"]].set_ylabel(plot["name"])

    # orientation special case
    axs[1, 1].plot(time, df["roll"], label="Roll")
    axs[1, 1].plot(time, df["pitch"], label="Pitch")
    axs[1, 1].plot(time, df["yaw"], label="Yaw")
    axs[1, 1].set_ylabel("Orientation (deg)")

    # Empty space
    axs[2, 1].axis("off")

    # time labels at the bottom
    axs[3, 0].set_xlabel("time (s)")
    axs[3, 0].set_xlabel("time (s)")

    for ax in axs.flat:
        if ax.has_data():
            ax.legend(loc="best", fontsize="small")
            ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
