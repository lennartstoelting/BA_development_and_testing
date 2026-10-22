import numpy as np
import pandas as pd
import os

from esekf_class import Filter

gravity_constant = 9.80665  # taken from scipy.constants.g
calibration_dt = 3  # length of timewindow at the start for IMU calibration in seconds

recordings_folder_path = "../csv_files_of_neon_recordings/"
unused_columns_in_recordings = [
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


def input_file_selection():
    try:
        input_files = [
            f for f in os.listdir(recordings_folder_path) if f.endswith(".csv")
        ]
        input_files.sort()
    except FileNotFoundError:
        print(f"Error: The directory '{recordings_folder_path}' does not exist.")
        return None

    if not input_files:
        print(
            f"The directory '{recordings_folder_path}' exists but no CSV files are found in the directory."
        )
        return None

    # User is prompted in terminal to select out of the available recordings
    print("\nThe following IMU data files are available:\n")
    for index, file in enumerate(input_files):
        print(f" {index} - {file}")

    selected_index_input = input("\nPlease select the input file by typing its index: ")

    try:
        selected_id = int(selected_index_input)
        if 0 <= selected_id < len(input_files):
            file_name = input_files[selected_id]
            print(f'Index provided, reading file:\n\n > "{file_name}"\n\n ============')
            return os.path.join(recordings_folder_path, file_name)
    except ValueError:
        pass

    default_file = input_files[0]
    print(
        f'Invalid selection! Defaulting to index 0, reading file:\n\n > "{default_file}"\n\n ============'
    )
    return os.path.join(recordings_folder_path, default_file)


def main():
    input_file_path = input_file_selection()
    try:
        all_imu_data = pd.read_csv(input_file_path, engine="python")
    except FileNotFoundError:
        print(f"Error: Reading '{input_file_path}' as CSV failed.")
        return
    imu_data = all_imu_data.drop(unused_columns_in_recordings, axis=1, errors="ignore")

    # ===
    # csv data retrieved and cleaned
    # ===

    start_time = imu_data["timestamp [ns]"].iloc[0] / 1e9  # nanoseconds to seconds
    previous_time = start_time  # previous_time is irrelevant in calibration steps
    esekf = Filter(calibration_dt)

    sum_of_corrections_triggered = 0
    for index, row in imu_data.iterrows():
        # prepare inputs
        current_time = row["timestamp [ns]"] / 1e9
        elapsed_time = current_time - start_time

        dt = current_time - previous_time

        accel_in_g = np.array([row[f"acceleration {i} [g]"] for i in ["x", "y", "z"]])
        gyro_in_deg_s = np.array([row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])

        # unit conversion
        a_tilde = accel_in_g * gravity_constant
        omega_tilde = gyro_in_deg_s * (np.pi / 180.0)

        # calibration
        if not esekf.calibration_complete:
            esekf.calibration_step(accel_in_g, omega_tilde, elapsed_time)
            esekf.input_and_state_history.append(
                np.hstack([None, a_tilde, omega_tilde, elapsed_time])
            )
            continue

        # filter is running
        esekf.prediction_step(a_tilde, omega_tilde, gravity_constant, dt)

        # this is probably only temporary, the I just want to do a check of whether to apply a correction step
        static_accel_threshold = 0.1
        static_gyro_threshold = 0.1
        accel_seems_static = (
            np.absolute(np.linalg.norm(a_tilde - esekf.x[10:13]) - gravity_constant)
            < static_accel_threshold
        )
        gyro_seems_static = (
            np.linalg.norm(omega_tilde - esekf.x[13:16]) < static_gyro_threshold
        )
        if accel_seems_static and gyro_seems_static:
            print(
                f"========================\ncorrection step now! (index {index}):\naccel: {accel_in_g}\ngyro: {gyro_in_deg_s}\n========================\n"
            )
        else:
            print(
                f"measurements don't seem static (index: {index}):\naccel {accel_in_g}\ngyro {gyro_in_deg_s}\n"
            )
            sum_of_corrections_triggered += 1
            esekf.correction_step(a_tilde, gravity_constant)

        # save state history
        esekf.input_and_state_history.append(
            np.hstack([esekf.x.copy(), accel_in_g, gyro_in_deg_s, elapsed_time])
        )

        # prepare for next iteration
        previous_time = current_time

    print(
        f"correction step triggered {sum_of_corrections_triggered} out of {imu_data.shape[0]} times"
    )

    esekf.plot_data(
        input_file_path.replace(recordings_folder_path, ""), state_history_output
    )


if __name__ == "__main__":
    main()
