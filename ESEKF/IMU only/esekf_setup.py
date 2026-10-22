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

output_columns = [
    "p_x",
    "p_y",
    "p_z",
    "v_x",
    "v_y",
    "v_z",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "a_b_x",
    "a_b_y",
    "a_b_z",
    "omega_b_x",
    "omega_b_y",
    "omega_b_z",
]


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
    esekf = Filter(imu_data, calibration_dt)

    for index, row in imu_data.iterrows():
        # prepare inputs
        current_time = row["timestamp [ns]"] / 1e9
        elapsed_time = current_time - start_time

        dt = current_time - previous_time

        accel_in_g = np.array([row[f"acceleration {i} [g]"] for i in ["x", "y", "z"]])
        gyro_in_deg = np.array([row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])

        # unit conversion
        a_tilde = accel_in_g * gravity_constant
        omega_tilde = gyro_in_deg * (np.pi / 180.0)

        # calibration
        if not esekf.calibration_complete:
            esekf.calibration_step(accel_in_g, omega_tilde, elapsed_time)
            continue

        # filter is running
        esekf.prediction_step(a_tilde, omega_tilde, gravity_constant, dt)

        # save state history
        esekf.state_history.append(np.append(esekf.x.copy(), elapsed_time))

        # das hier ist gerade so der große Punkt, Ich will das nicht in dem bereits bestehenden array an die richtige Stelle machen sondern so wie bei der nur state history das immer im loop machen
        # einziges Problem da ist, das der loop ja schon früher continued wird, das sollte ich also beachten.
        esekf.input_and_state_history.iloc[
            index, esekf.input_and_state_history.columns.get_indexer(output_columns)
        ] = esekf.x.flatten()

        # prepare for next iteration
        previous_time = current_time

    esekf.plot_data()


if __name__ == "__main__":
    main()
