"""
Debug plots for the filter history CSV.

Structure:
- a *panel* is any function `panel(ax, df)` that draws one subplot
- a *layout* is a list of rows, each row a list of panels (None = leave empty)
- `plot_grid(layout, ...)` builds the figure from a layout

New grid:        write a new layout and pass it to plot_grid
New kind of plot: write a new panel function (or panel factory)
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation

from config import FILTER_OUTPUT_FILE
from modules import get_def

COMPONENT_COLORS = ["red", "green", "blue"]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def _load_history(history_file):
    """Read the filter history CSV and add Euler angles (deg) as extra columns."""
    df = pd.read_csv(history_file)

    quaternion_columns = get_def("IMU orientation")["dims"]
    rotations = Rotation.from_quat(df[quaternion_columns].values, scalar_first=True)
    euler = rotations.as_euler("yxz", degrees=True)
    df["roll"], df["pitch"], df["yaw"] = euler[:, 0], euler[:, 1], euler[:, 2]

    return df


# ---------------------------------------------------------------------------
# panels: each one draws a single subplot
# ---------------------------------------------------------------------------


def vector_panel(name):
    """x/y/z components of a quantity known to get_def, e.g. "IMU velocity"."""

    def draw(ax, df):
        for symbol, color in zip(get_def(name)["dims"], COMPONENT_COLORS):
            ax.plot(df["time"], df[symbol], label=f"${symbol}$", color=color)
        ax.set_ylabel(name)

    return draw


def orientation_panel(angles=("roll", "pitch", "yaw")):
    """Euler angles of the orientation, pick which ones to show."""

    def draw(ax, df):
        for angle in angles:
            ax.plot(df["time"], df[angle], label=angle.capitalize())
        ax.set_ylabel("Orientation (deg)")

    return draw


def norm_panel(norm_column, threshold_column):
    """A norm signal together with its threshold."""

    def draw(ax, df):
        ax.plot(df["time"], df[norm_column], label=f"|{norm_column}|", color="yellow")
        ax.plot(df["time"], df[threshold_column], label=threshold_column, color="black")

    return draw


def covariance_panel(ax, df):
    ax.plot(df["time"], df["covariance_temp"], label="det P")
    ax.set_ylabel("log scale")


# ---------------------------------------------------------------------------
# generic grid builder
# ---------------------------------------------------------------------------


def plot_grid(layout, title, figsize=(16, 16)):
    df = _load_history(FILTER_OUTPUT_FILE)

    n_rows = len(layout)
    n_cols = max(len(row) for row in layout)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title)

    for row_id, panels in enumerate(layout):
        for col_id in range(n_cols):
            ax = axs[row_id, col_id]
            panel = panels[col_id] if col_id < len(panels) else None

            if panel is None:
                ax.axis("off")
                continue

            if isinstance(panel, str):
                panel = vector_panel(panel)

            panel(ax, df)

            if row_id == n_rows - 1:
                ax.set_xlabel("time (s)")
            if ax.has_data():
                ax.legend(loc="best", fontsize="small")
                ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
