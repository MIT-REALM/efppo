import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from efppo.task.dyn_types import BTState
from efppo.task.task import Task
from efppo.utils.register_cmaps import register_cmaps


def C(ii: int):
    colors = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"]
    return colors[ii % len(colors)]


class Plotter:
    def __init__(self, task: Task, rel_path: pathlib.Path | None = None, dpi: int = None):
        self.task = task
        self.rel_path = rel_path
        self.dpi = dpi
        register_cmaps()

    def plot_traj(self, bT_state: BTState, multicolor: bool = False, ax: plt.Axes = None):
        bT_x, bT_y = self.task.get2d(bT_state)
        bT_line = np.stack([bT_x, bT_y], axis=-1)

        if ax is None:
            fig, ax = plt.subplots(dpi=self.dpi)
        else:
            fig = ax.figure

        self.task.setup_traj_plot(ax)
        colors = C(1)
        if multicolor:
            colors = [C(ii) for ii in range(bT_line.shape[1])]
        line_col = LineCollection(bT_line, lw=1.0, zorder=5, colors=colors)
        ax.add_collection(line_col)

        # Starts and Ends.
        ax.scatter(bT_x[:, 0], bT_y[:, 0], color="black", s=1**2, zorder=6, marker="s")
        ax.scatter(bT_x[:, -1], bT_y[:, -1], color="green", s=1**2, zorder=7, marker="o")

        ax.autoscale_view()
        return fig
