import numpy as np
import pandas as pd
import os

from esekf_class import Filter

gravity_constant = 9.80665  # taken from scipy.constants.g
calibration_dt = 3  # length of timewindow at the start for IMU calibration in seconds


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

    print("\nThe following IMU data files are available:\n")
    for index, file in enumerate(input_files):
        print(f" {index} - {file}")

    selected_index_input = input("\nPlease select the input file by typing its index: ")

    try:
        selected_id = int(selected_index_input)
        if 0 <= selected_id < len(input_files):
            file_name = input_files[selected_id]
            print(f'Index provided, reading file:\n\n > "{file_name}"\n\n ============')
            return os.path.join(folder_path, file_name)
    except ValueError:
        pass

    default_file = input_files[0]
    print(
        f'Invalid selection! Defaulting to index 0, reading file:\n\n > "{default_file}"\n\n ============'
    )
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

    start_time = imu_data["timestamp [ns]"].iloc[0] / 1e9  # nanoseconds to seconds
    # this is technically not correct but for now, I am not using previous_time until the calibration is finished so there is no conflict for now
    previous_time = start_time

    esekf = Filter(calibration_dt)

    for index, row in imu_data.iterrows():

        # prepare inputs
        current_time = row["timestamp [ns]"] / 1e9
        delta_t = current_time - previous_time
        elapsed_time = current_time - start_time

        accel_in_g = np.array(
            [
                row["acceleration x [g]"],
                row["acceleration y [g]"],
                row["acceleration z [g]"],
            ]
        )
        a_tilde = accel_in_g * gravity_constant
        gyro_in_deg = np.array(
            [row["gyro x [deg/s]"], row["gyro y [deg/s]"], row["gyro z [deg/s]"]]
        )
        omega_tilde = gyro_in_deg * (np.pi / 180.0)

        if not esekf.calibration_complete:
            esekf.calibration_step(accel_in_g, omega_tilde, elapsed_time)
        previous_time = current_time


if __name__ == "__main__":
    main()
