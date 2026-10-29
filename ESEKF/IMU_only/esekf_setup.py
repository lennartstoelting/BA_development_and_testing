import numpy as np
import pandas as pd
import os

from esekf_class import Filter
from esekf_definitions import *


recordings_folder = "../csv_files_of_neon_recordings/"
unused_cols_of_recordings = [
    "roll [deg]",
    "pitch [deg]",
    "yaw [deg]",
    "quaternion x",
    "quaternion y",
    "quaternion z",
    "quaternion w",
    "recording id",
]

state_history_output = "IMU_input_and_states_over_time.csv"


def main():
    try:
        input_file_path = input_file_selection()
        imu_data = get_IMU_data(input_file_path)
    except FileNotFoundError:
        return

    # ===
    # timers for loop
    # ===

    start_time = imu_data["timestamp [ns]"].iloc[0] / 1e9  # nanoseconds to seconds
    previous_time = start_time  # previous_time is irrelevant in calibration steps
    esekf = Filter(calibration_dt)

    corr_step_triggered = 0
    for index, row in imu_data.iterrows():
        current_time = row["timestamp [ns]"] / 1e9
        elapsed_time = current_time - start_time
        dt = current_time - previous_time

        # ===
        # sensor inputs
        # ===

        accel_in_g = np.array([row[f"acceleration {i} [g]"] for i in ["x", "y", "z"]])
        gyro_in_deg_s = np.array([row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])

        # unit conversion
        accel_in_m_ss = accel_in_g * gravity_constant
        gyro_in_rad_s = gyro_in_deg_s * (np.pi / 180.0)

        # ===
        # calibration step
        # ===

        if not esekf.calibration_complete:
            esekf.calibration_step(accel_in_g, gyro_in_rad_s, elapsed_time)
            esekf.input_and_state_history.append(
                np.hstack([None, accel_in_m_ss, gyro_in_rad_s, elapsed_time])
            )
            continue

        # ===
        # prediction and measurement step
        # ===

        # esekf.prediction_step(accel_in_m_ss, gyro_in_rad_s, gravity_constant, dt)

        # static_accel_threshold = 0.1
        # static_gyro_threshold = 0.1
        # accel_seems_static = (
        #     np.absolute(
        #         np.linalg.norm(
        #             accel_in_m_ss - esekf.x[get_def("IMU accel bias")["slice"]]
        #         )
        #         - gravity_constant
        #     )
        #     < static_accel_threshold
        # )
        # gyro_seems_static = (
        #     np.linalg.norm(gyro_in_rad_s - esekf.x[get_def("IMU accel bias")["slice"]])
        #     < static_gyro_threshold
        # )
        # if accel_seems_static and gyro_seems_static:
        #     # print(
        #     #     f"========================\ncorrection step now! (index {index}):\naccel: {accel_in_g}\ngyro: {gyro_in_deg_s}\n========================\n"
        #     # )
        #     esekf.correction_step(accel_in_m_ss, gravity_constant)
        # else:
        #     corr_step_triggered += 1
        #     # print(
        #     #     f"measurements don't seem static (index: {index}):\naccel {accel_in_g}\ngyro {gyro_in_deg_s}\n"
        #     # )

        # save state history
        esekf.input_and_state_history.append(
            np.hstack([esekf.x.copy(), accel_in_g, gyro_in_deg_s, elapsed_time])
        )

        # prepare for next iteration
        previous_time = current_time

    print(
        f"correction step triggered: {corr_step_triggered} / {imu_data.shape[0]} times"
    )
    print(f"determinant of covariance P at the end: {np.linalg.det(esekf.P)}")

    esekf.plot_data(
        input_file_path.replace(recordings_folder, ""), state_history_output
    )


def input_file_selection():
    try:
        input_files = [f for f in os.listdir(recordings_folder) if f.endswith(".csv")]
    except FileNotFoundError:
        print(f"Error: The directory '{recordings_folder}' does not exist.")
        raise FileNotFoundError

    if not input_files:
        print(f"The directory '{recordings_folder}' exists but contains no CSV files.")
        raise FileNotFoundError

    # User is prompted in terminal to select out of the available recordings
    print("\nThe following IMU data files are available:\n")
    [print(f" {index} - {file}") for index, file in enumerate(input_files)]
    selected_index_input = input("\nPlease select the input file by typing its index: ")

    file_name = None

    if selected_index_input == "q":
        print("aborting script execution")
        raise ValueError

    try:
        selected_id = int(selected_index_input)
        if 0 <= selected_id < len(input_files):
            file_name = input_files[selected_id]
            print(f"reading file:")
    except ValueError:
        pass

    if not file_name:
        print(f"Invalid selection! Defaulting to index 0, reading file:")
        file_name = input_files[0]

    print(f'\n > "{file_name}"\n\n ============')
    return os.path.join(recordings_folder, file_name)


def get_IMU_data(file):
    try:
        all_imu_data = pd.read_csv(file, engine="python")
    except FileNotFoundError:
        print(f"Error: Reading '{file}' as CSV failed.")
        raise FileNotFoundError

    return all_imu_data.drop(unused_cols_of_recordings, axis=1, errors="ignore")


if __name__ == "__main__":
    main()
