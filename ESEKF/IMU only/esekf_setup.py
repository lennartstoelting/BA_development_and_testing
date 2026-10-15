import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants


def input_file_selection():
    path_prefix = "../csv_files_of_neon_recordings/"
    input_files = [
        "imu_2+ min of no motion.csv",
        "imu_1 min static with one moving person.csv",
        "imu_Moving camera + moving person.csv",
        "imu_Narural head movement with person in the scene.csv",
        "imu_Static camera, person with tiny movement.csv",
    ]
    full_input_path = lambda index=0: path_prefix + input_files[index]

    # ---

    print("The following IMU data files are available:\n")
    for index, file in enumerate(input_files):
        print(f"{index} - {file}")
    selected_index_input = input("\nPlease select the input file by typing its index: ")

    default_message = f'-> Defaulting to index 0, reading file "{input_files[0]}"...'
    try:
        selected_id = int(selected_index_input)
    except ValueError:
        print(f"The chosen index could not be converted to integer.\n{default_message}")
        return full_input_path(0)

    if selected_id < 0 or selected_id >= len(input_files):
        print(f"The chosen index is negative or too large.\n{default_message}")
        return full_input_path(0)

    print(f'Index provided, reading file "{input_files[selected_id]}"...')
    return full_input_path(selected_id)


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

    # esekf = Filter(calibration_samples, imu_data)

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
