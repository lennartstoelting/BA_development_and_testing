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


def def_vector_panel(name):
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


def panel(*series, ylabel=None):
    """
    Plot one or more history columns against time.

    Each entry in `series` is a (column, label) tuple.
    """

    def draw(ax, df):
        for i, item in enumerate(series):
            column, label = item
            ax.plot(df["time"], df[column], label=label, color=COMPONENT_COLORS[i])
        if ylabel:
            ax.set_ylabel(ylabel)

    return draw


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
            panel_func = panels[col_id] if col_id < len(panels) else None

            if panel_func is None:
                ax.axis("off")
                continue

            if isinstance(panel_func, str):
                panel_func = def_vector_panel(panel_func)

            panel_func(ax, df)

            if row_id == n_rows - 1:
                ax.set_xlabel("time (s)")
            if ax.has_data():
                ax.legend(loc="best", fontsize="small")
                ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.canvas.manager.full_screen_toggle()
    plt.show()
