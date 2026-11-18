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


COLUMNS_TO_KEEP_FROM_RAW_DATA = [
    # "recording id",
    "timestamp [ns]",
    "acceleration x [g]",
    "acceleration y [g]",
    "acceleration z [g]",
    "gyro x [deg/s]",
    "gyro y [deg/s]",
    "gyro z [deg/s]",
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
        all_imu_data = load_recording(selected_IMU_file)
        imu_data = all_imu_data[COLUMNS_TO_KEEP_FROM_RAW_DATA]
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    timer = Timer()
    esekf = Filter()

    print("running filter...")
    for index, row in imu_data.iterrows():
        esekf.process_IMU_sample(row, index)
    print(f"filter finished {timer.check()}")
    del timer

    # plot results
    save_filter_history(esekf.filter_history)
    plot_filter_history(selected_IMU_file.name)


# ---------------------------------------------------------------------------
# finding, selecting and loading IMU recording file
# ---------------------------------------------------------------------------


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

    print(f'\n > "{selected_file.name}"\n\n ============\n')
    return selected_file


def load_recording(file):
    try:
        return pd.read_csv(file, engine="python")
    except FileNotFoundError:
        raise FileNotFoundError(f"Reading '{file}' as CSV failed.")


# ---------------------------------------------------------------------------
# saving and plotting finished filter
# ---------------------------------------------------------------------------


def save_filter_history(history_array):

    print("\n ============\n")
    print(" saving filter results to csv...")

    if not history_array:
        print("Combined history of inputs and states is empty, can't be plotted")
        return

    df = pd.DataFrame(
        history_array,
        columns=[
            *[
                dim
                for comp in get_def("IMU nominal state")["components"]
                for dim in get_def(comp["name"])["dims"]
            ],  # state
            *[
                dim
                for comp in get_def("IMU input")["components"]
                for dim in get_def(comp["name"])["dims"]
            ],  # raw IMU sensor measurements (units converted)
            *["accel_norm", "accel_threshold"],  # accel static check
            *["gyro_norm", "gyro_threshold"],  # gyro static check
            "covariance_test",  # det P
            *["neon_roll", "neon_pitch", "neon_yaw"],
            "time",
        ],
    )
    FILTER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTER_OUTPUT_FILE, index=False)


def plot_filter_history(title):

    print("\n ============\n")
    print(" plotting filter results from csv (press q in plot window to exit)...")

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
            [orientation_panel(angles=("roll", "pitch", "yaw"))],
            [orientation_panel(angles=("neon_roll", "neon_pitch", "neon_yaw"))],
        ],
    }

    for name, layout in plots.items():
        if name in (
            "input and states with covariance",
            "correction trigger accel and gyro norm",
            "orientation closeup",
        ):
            plot_grid(layout, f"file: {title}\n{name}")


if __name__ == "__main__":
    main()
