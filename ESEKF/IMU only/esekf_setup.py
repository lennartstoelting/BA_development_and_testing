import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants


import os


def input_file_selection():
    folder_path = "../csv_files_of_neon_recordings/"
    try:
        input_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        input_files.sort()
    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' does not exist.")
        return None

    if not input_files:
        print("No CSV files found in the directory.")
        return None

    print("The following IMU data files are available:\n")
    for index, file in enumerate(input_files):
        print(f"{index} - {file}")

    selected_index_input = input("\nPlease select the input file by typing its index: ")

    try:
        selected_id = int(selected_index_input)
        if 0 <= selected_id < len(input_files):
            file_name = input_files[selected_id]
            print(f'Index provided, reading file "{file_name}"...')
            return os.path.join(folder_path, file_name)
    except ValueError:
        pass

    default_file = input_files[0]
    print(f'Invalid selection. Defaulting to index 0, reading file "{default_file}"...')
    return os.path.join(folder_path, default_file)


def main():
    input_file_path = input_file_selection()

    try:
        all_imu_data = pd.read_csv(input_file_path, engine="python")
    except FileNotFoundError:
        print(f"Error: Could not find '{input_file_path}'.")
        return

    imu_data = all_imu_data.drop(
        [
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
            "quaternion x",
            "quaternion y",
            "quaternion z",
            "quaternion w",
        ],
        axis=1,
        errors="ignore",
    )

    start_time = imu_data["timestamp [ns]"].iloc[0]
    previous_time = start_time

    esekf = Filter(calibration_samples, imu_data)

    for index, row in imu_data.iterrows():

        # prepare inputs
        current_time = row["timestamp [ns]"]
        delta_t = (current_time - previous_time) / 1e9
        elapsed_time = (current_time - start_time) / 1e9

        gyro = np.array(
            [row["gyro x [deg/s]"], row["gyro y [deg/s]"], row["gyro z [deg/s]"]]
        )
        accel = np.array(
            [
                row["acceleration x [g]"],
                row["acceleration y [g]"],
                row["acceleration z [g]"],
            ]
        )
        previous_time = current_time


if __name__ == "__main__":
    main()
