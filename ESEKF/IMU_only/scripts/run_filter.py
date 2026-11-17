"""
The bridge between
- the user
- the recording files
- the filter instance
- implemented theory
- plotting tools

This file should not contain any of the actual mathmatical operations being performed by the filter,
it's only the connection between processes
"""

import numpy as np
import pandas as pd
import os
import sys

# go to parent directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import *
from modules.filter_class import Filter
from scripts.plot_filter_results import *


rawIMU_recs_path = "../../csv_files_of_neon_recordings/"

filter_output_filename = "IMU_input_and_states_over_time.csv"
filter_output_path = "../out/"
finished_filter_output_path = filter_output_path + filter_output_filename


def main():
    # fetch recording files names, ask user which one to run and prune unused data columns
    try:
        list_of_rawIMU_csv_filenames = gather_IMU_files()
        input_file_path = prompt_file_selection(list_of_rawIMU_csv_filenames)
        print(input_file_path)
        all_imu_data = read_csv(input_file_path)
        imu_data = clean_csv(all_imu_data)
    except (FileNotFoundError, ValueError):
        return

    # run filter
    esekf = Filter()
    # print(f"\ndeterminant of covariance P at the start: {np.linalg.det(esekf.P)}")
    for index, row in imu_data.iterrows():
        esekf.process_IMU_sample(row, index)

    # print(f"\ndeterminant of covariance P at the end: {np.linalg.det(esekf.P)}")

    # plot results
    plot_data(
        esekf.input_and_state_history,
        # "../recordings/IMU.csv" -> "IMU.csv"
        input_file_path.replace(rawIMU_recs_path, ""),
        finished_filter_output_path,
    )


def gather_IMU_files():
    try:
        list_of_rawIMU_csv_filenames = [
            f for f in os.listdir(rawIMU_recs_path) if f.endswith(".csv")
        ]
    except FileNotFoundError:
        print(f"Error: The directory '{rawIMU_recs_path}' does not exist.")
        raise FileNotFoundError

    if not list_of_rawIMU_csv_filenames:
        print(f"The directory '{rawIMU_recs_path}' exists but contains no CSV files.")
        raise FileNotFoundError

    return list_of_rawIMU_csv_filenames


def prompt_file_selection(list_of_rawIMU_csv_filenames):
    # User is prompted in terminal to select out of the available recordings
    print("\nThe following IMU data files are available:\n")
    [
        print(f" {index} - {file}")
        for index, file in enumerate(list_of_rawIMU_csv_filenames)
    ]
    print(f" q - abort script")
    selected_index_input = input("\nPlease select the input file by typing its index: ")

    file_name = None

    if selected_index_input == "q":
        print("\naborted script execution\n")
        raise ValueError

    try:
        selected_id = int(selected_index_input)
        if 0 <= selected_id < len(list_of_rawIMU_csv_filenames):
            file_name = list_of_rawIMU_csv_filenames[selected_id]
            print(f"reading file:")
    except ValueError:
        pass

    if not file_name:
        print(f"Invalid selection! Defaulting to index 0, reading file:")
        file_name = list_of_rawIMU_csv_filenames[0]

    print(f'\n > "{file_name}"\n\n ============')
    return os.path.join(rawIMU_recs_path, file_name)


def read_csv(file):
    try:
        return pd.read_csv(file, engine="python")
    except FileNotFoundError:
        print(f"Error: Reading '{file}' as CSV failed.")
        raise FileNotFoundError


def clean_csv(all_imu_data):
    unused_cols_of_recordings = [
        "recording id",
        # "roll [deg]",
        # "pitch [deg]",
        # "yaw [deg]",
        # "quaternion x",
        # "quaternion y",
        # "quaternion z",
        # "quaternion w",
    ]

    return all_imu_data.drop(unused_cols_of_recordings, axis=1, errors="ignore")


def plot_data(history_array, input_file_path, output):
    if not history_array:
        print("Combined history of inputs and states is empty, can't be plotted")
        return

    df = pd.DataFrame(
        history_array,
        columns=[
            *get_def("IMU position")["dims"],
            *get_def("IMU velocity")["dims"],
            *get_def("IMU orientation")["dims"],
            *get_def("IMU accel bias")["dims"],
            *get_def("IMU gyro bias")["dims"],
            *get_def("IMU accelerometer")["dims"],
            *get_def("IMU gyroscope")["dims"],
            "accel_norm",
            "accel_threshold",
            "gyro_norm",
            "gyro_threshold",
            "covariance_temp",
            "time",
        ],
    )
    df.to_csv(output, index=False)

    # pyplot_input_grid(output, input_file_path)
    # pyplot_input_and_state_grid(output, input_file_path)
    # pyplot_orientation_grid(output, input_file_path)


if __name__ == "__main__":
    main()
