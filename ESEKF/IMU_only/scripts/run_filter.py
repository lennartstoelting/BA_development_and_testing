"""
The bridge between
- the user
- the recording files
- the filter instance
- implemented theory
- plotting tools

This file should not contain any of the actual mathematical operations being performed by the filter,
it's only the connection between processes
"""

import pandas as pd

from config import RAW_IMU_RECS_DIR, FILTER_OUTPUT_FILE  # must come first
from modules import *
from modules.filter_class import Filter
from modules.plot_filter_results import *

# from plot_filter_results import pyplot_orientation_grid

UNUSED_RECORDING_COLUMNS = [
    "recording id",
    "roll [deg]",
    "pitch [deg]",
    "yaw [deg]",
    # "quaternion x",
    # "quaternion y",
    # "quaternion z",
    # "quaternion w",
]


def main():
    # find recordings, ask user which one to run and prune unused data columns
    try:
        recording_files = find_recording_files()
        selected_IMU_file = prompt_file_selection(recording_files)
        imu_data = drop_unused_columns(load_recording(selected_IMU_file))
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    # run filter
    esekf = Filter()
    for index, row in imu_data.iterrows():
        esekf.process_IMU_sample(row, index)

    # plot results
    save_filter_history(esekf.input_and_state_history)
    plot_filter_history(selected_IMU_file.name)


def find_recording_files() -> list:
    if not RAW_IMU_RECS_DIR.is_dir():
        raise FileNotFoundError(f"The directory '{RAW_IMU_RECS_DIR}' does not exist.")

    recording_files = sorted(RAW_IMU_RECS_DIR.glob("*.csv"))
    if not recording_files:
        raise FileNotFoundError(
            f"The directory '{RAW_IMU_RECS_DIR}' exists but contains no CSV files."
        )
    return recording_files


def prompt_file_selection(recording_files):
    print("\nThe following IMU data files are available:\n")
    for index, file in enumerate(recording_files):
        print(f" {index} - {file.name}")
    print(" q - abort script")
    answer = input("\nPlease select the input file by typing its index: ")

    if answer == "q":
        raise ValueError("\naborted script execution\n")

    try:
        selected_file = recording_files[int(answer)]
        print("reading file:")
    except (ValueError, IndexError):
        selected_file = recording_files[0]
        print("Invalid selection! Defaulting to index 0, reading file:")

    print(f'\n > "{selected_file.name}"\n\n ============')
    return selected_file


def load_recording(file):
    try:
        return pd.read_csv(file, engine="python")
    except FileNotFoundError:
        raise FileNotFoundError(f"Reading '{file}' as CSV failed.")


def drop_unused_columns(all_imu_data):
    return all_imu_data.drop(UNUSED_RECORDING_COLUMNS, axis=1, errors="ignore")


def save_filter_history(history_array):
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
            *["accel_norm", "accel_threshold"],
            *["gyro_norm", "gyro_threshold"],
            "covariance_test",
            "time",
        ],
    )
    FILTER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTER_OUTPUT_FILE, index=False)


def plot_filter_history(title):
    plots = {
        # fmt: off
        "input and states with covariance": [
            ["IMU accelerometer", "IMU gyroscope"],
            ["IMU velocity", orientation_panel()],
            ["IMU position", panel(("covariance_test", "det P"), ylabel="log scale"),],
            ["IMU accel bias", "IMU gyro bias"],
        ],
        # fmt: on
        "correction trigger accel and gyro norm": [
            ["IMU accelerometer", "IMU gyroscope"],
            [
                panel(
                    ("accel_threshold", "accel_threshold"),
                    ("accel_norm", r"$|(\|a\| - g)|$"),
                ),
                panel(
                    ("gyro_threshold", "gyro_threshold"),
                    ("gyro_norm", r"$\|\omega\|$"),
                ),
            ],
        ],
        "orientation closeup": [
            ["IMU accelerometer"],
            [orientation_panel(angles=("roll", "pitch", "yaw"))],
        ],
    }

    for name, layout in plots.items():
        if name in (
            "input and states with covariance",
            "correction trigger accel and gyro norm",
            # "orientation closeup",
        ):
            plot_grid(layout, f"file: {title}\n{name}")


if __name__ == "__main__":
    main()
