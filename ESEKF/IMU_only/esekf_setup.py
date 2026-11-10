"""
The bridge between
- the user
- the recording files
- the filter instance
- implemented theory
- plotting tools

This file should not contain any of the actual mathmatical operations being performed by the filter,
it's only the middleman
"""

import numpy as np
import pandas as pd
import os

from esekf_class import Filter
from esekf_definitions import *
from esekf_plot import *


recordings_folder = "../csv_files_of_neon_recordings/"

state_history_output = "IMU_input_and_states_over_time.csv"


def main():
    # fetch recording files names, ask user which one to run and prune unused data columns
    try:
        input_files = gather_IMU_files()
        input_file_path = prompt_file_selection(input_files)
        all_imu_data = read_csv(input_file_path)
        imu_data = clean_csv(all_imu_data)
    except (FileNotFoundError, ValueError):
        return

    # run filter
    esekf = Filter()
    for index, row in imu_data.iterrows():
        esekf.process_IMU_sample(row)

    # plot results
    plot_data(
        esekf.input_and_state_history,
        input_file_path.replace(recordings_folder, ""),
        state_history_output,
    )


def gather_IMU_files():
    try:
        input_files = [f for f in os.listdir(recordings_folder) if f.endswith(".csv")]
    except FileNotFoundError:
        print(f"Error: The directory '{recordings_folder}' does not exist.")
        raise FileNotFoundError

    if not input_files:
        print(f"The directory '{recordings_folder}' exists but contains no CSV files.")
        raise FileNotFoundError

    return input_files


def prompt_file_selection(input_files):
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


def read_csv(file):
    try:
        return pd.read_csv(file, engine="python")
    except FileNotFoundError:
        print(f"Error: Reading '{file}' as CSV failed.")
        raise FileNotFoundError


def clean_csv(all_imu_data):
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
            *get_def("IMU accelerometer (m/s^2)")["dims"],
            *get_def("IMU gyroscope (rad/s)")["dims"],
            "time",
        ],
    )
    df.to_csv(output, index=False)

    pyplot_input_and_state_grid(output, input_file_path)


if __name__ == "__main__":
    main()
