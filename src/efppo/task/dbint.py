import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import shapely
from matplotlib import pyplot as plt
from typing_extensions import override

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State
from efppo.task.task import Task, TaskState
from efppo.utils.constr_utils import box_constraint
from efppo.utils.jax_types import BBFloat
from efppo.utils.plot_utils import plot_x_bounds, plot_x_goal, plot_y_bounds, poly_to_patch
from efppo.utils.rng import PRNGKey


class DbInt(Task):
    NX = 2
    NU = 1

    def __init__(self):
        self.goal_x = 0.75
        self.goal_r = 0.1
        self.pos_wall = 1.0
        self.dt = 0.025

    @override
    def step(self, state: State, control: Control) -> State:
        p, v = self.chk_x(state)
        control = self.discr_to_cts(control)
        (a,) = self.chk_u(control)
        # Integrate exactly.
        p_new = p + v * self.dt + 0.5 * a * self.dt**2
        v_new = v + a * self.dt
        return jnp.array([p_new, v_new])

    def discr_to_cts(self, control_idx) -> Control:
        """Convert discrete control to continuous."""
        return jnp.array(self.discrete_actions)[control_idx]

    @property
    def discrete_actions(self):
        actions = np.array([[-1.0], [0.0], [1.0]])
        assert actions.shape == (3, 1)
        return actions

    @property
    def n_actions(self) -> int:
        return len(self.discrete_actions)

    @override
    def get_obs(self, state: State) -> Obs:
        return self.chk_x(state)

    def sample_x0_train(self, key: PRNGKey, num: int) -> TaskState:
        xmin = np.array([-1.5, -1.5])
        xmax = np.array([1.5, 1.5])
        return jr.uniform(key, shape=(num, self.nx), minval=xmin, maxval=xmax)

    def grid_contour(self) -> tuple[BBFloat, BBFloat, TaskState]:
        b_p = np.linspace(-1.5, 1.5, num=64)
        b_v = np.linspace(-1.5, 1.5, num=64)
        bb_X, bb_Y = np.meshgrid(b_p, b_v)
        bb_state = np.stack([bb_X, bb_Y], axis=-1)
        return bb_X, bb_Y, bb_state

    def get_x0_eval(self, num: int = 32) -> TaskState:
        with jax.ensure_compile_time_eval():
            key = jr.PRNGKey(1234567)
            b_x0 = self.sample_x0_train(key, num)
            b_in_ci = jax.vmap(self.in_ci)(b_x0)
            b_x0 = b_x0[b_in_ci]
        return b_x0

    def in_ci(self, state: State):
        x, v = state
        in_right = x <= 1 - jnp.maximum(v, 0) ** 2 / 2
        in_left = x >= jnp.minimum(v, 0) ** 2 / 2 - 1
        within_v = jnp.abs(v) <= 1

        return in_left & in_right & within_v

    @property
    def x_labels(self) -> list[str]:
        return ["$p$", "$v$"]

    @property
    def u_labels(self) -> list[str]:
        return ["$a$"]

    @property
    def l_labels(self) -> list[str]:
        return ["pos_cost"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$p_l$", r"$p_u$", r"$v_l$", r"$v_u$"]

    def l_components(self, state: State, control: Control) -> LFloat:
        """Quadratic cost on position and control."""
        p, v = self.chk_x(state)

        # Distance to goal_x +- goal_r.
        dist = jax.nn.relu(jnp.abs(p - self.goal_x) - self.goal_r)
        pos_cost = jnp.tanh(dist)
        return jnp.array([pos_cost])

    def l(self, state: State, control: Control) -> LFloat:
        weights = jnp.array([0.05])
        l_components = self.l_components(state, control)
        return jnp.sum(l_components * weights)

    def h_components(self, state: State) -> HFloat:
        """Hard constraints on position and velocity. Negative is safe."""
        p, v = self.chk_x(state)
        p_l, p_u = box_constraint(p, -1.0, +1.0)
        v_l, v_u = box_constraint(v, -1.0, +1.0)
        h_h = jnp.array([p_l, p_u, v_l, v_u])
        h_h = jnp.where(h_h >= 0, h_h + 0.5, h_h - 0.5)
        return h_h

    def get_ci_points(self):
        all_xs, all_vs = [], []
        vs = np.linspace(-1.0, 1.0)
        xs = self.pos_wall - np.maximum(vs, 0.0) ** 2 / 2
        all_xs += [xs]
        all_vs += [vs]

        vs = np.linspace(-1.0, 1.0)[::-1]
        xs = np.minimum(vs, 0.0) ** 2 / 2 - self.pos_wall
        all_xs += [xs]
        all_vs += [vs]

        all_xs, all_vs = np.concatenate(all_xs), np.concatenate(all_vs)
        assert all_xs.ndim == all_vs.ndim == 1

        return np.stack([all_xs, all_vs], axis=1)

    def setup_traj_plot(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = xlim = [-1.5, 1.5]
        PLOT_YMIN, PLOT_YMAX = ylim = [-1.5, 1.5]
        ax.set_facecolor("0.98")
        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel="Position", ylabel="Velocity")

        # 2: Plot the avoid set
        obs_style = dict(facecolor="0.6", edgecolor="none", alpha=0.4, zorder=3, hatch="/")
        plot_x_bounds(ax, (-1.0, 1.0), obs_style)
        plot_y_bounds(ax, (-1.0, 1.0), obs_style)

        # 3: Plot the goal set.
        goal_style = dict(facecolor="green", edgecolor="none", alpha=0.3, zorder=4.0)
        plot_x_goal(ax, (self.goal_x - self.goal_r, self.goal_x + self.goal_r), goal_style)

        # 4: Plot the CI.
        outside_pts = [(PLOT_XMIN, PLOT_YMIN), (PLOT_XMIN, PLOT_YMAX), (PLOT_XMAX, PLOT_YMAX), (PLOT_XMAX, PLOT_YMIN)]
        outside = shapely.Polygon(outside_pts)

        ci_pts = self.get_ci_points()
        hole = shapely.Polygon(ci_pts)

        ci_poly = outside.difference(hole)
        patch = poly_to_patch(ci_poly, facecolor="0.6", edgecolor="none", alpha=0.5, zorder=3)
        ax.add_patch(patch)
        hatch_color = "0.5"
        patch = poly_to_patch(ci_poly, facecolor="none", edgecolor=hatch_color, linewidth=0, zorder=3.1, hatch=".")
        ax.add_patch(patch)
