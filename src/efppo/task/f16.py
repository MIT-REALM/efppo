import itertools

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax_f16.f16 import F16
from matplotlib import pyplot as plt
from typing_extensions import override

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State
from efppo.task.f16_safeguarded import MORELLI_BOUNDS, F16Safeguarded
from efppo.task.task import Task, TaskState
from efppo.utils.angle_utils import rotx, roty, rotz
from efppo.utils.jax_types import BBFloat, BoolScalar, FloatScalar
from efppo.utils.jax_utils import box_constr_clipmax, box_constr_log1p, merge01, tree_add, tree_inner_product, tree_mac
from efppo.utils.plot_utils import plot_x_bounds, plot_y_bounds, plot_y_goal
from efppo.utils.rng import PRNGKey


def ode4(xdot, dt: float, state):
    """RK4."""
    state0 = state
    k0 = xdot(state0)

    state1 = tree_mac(state0, 0.5 * dt, k0)
    k1 = xdot(state1)

    state2 = tree_mac(state0, 0.5 * dt, k1)
    k2 = xdot(state2)

    state3 = tree_mac(state0, dt, k2)
    k3 = xdot(state3)

    # state1 = state0 + dt * (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
    coefs = np.array([1.0, 2.0, 2.0, 1.0]) * dt / 6.0
    state_out = tree_add(state0, tree_inner_product(coefs, [k0, k1, k2, k3]))
    return state_out


def discretize_u(controls: list[list[float]]):
    """Discretize controls by doing a Cartesian product."""
    controls = list(itertools.product(*controls))
    return np.stack(controls, axis=0)


def compute_f16_vel_angles(state: State):
    """Compute cos / sin of [gamma, sigma], the pitch & yaw of the velocity vector."""
    assert state.shape == (F16.NX,)
    # 1: Compute {}^{W}R^{F16}.
    R_W_F16 = rotz(state[F16.PSI]) @ roty(state[F16.THETA]) @ rotx(state[F16.PHI])
    assert R_W_F16.shape == (3, 3)

    # 2: Compute v_{F16}
    ca, sa = jnp.cos(state[F16.ALPHA]), jnp.sin(state[F16.ALPHA])
    cb, sb = jnp.cos(state[F16.BETA]), jnp.sin(state[F16.BETA])
    v_F16 = jnp.array([ca * cb, sb, sa * cb])

    # 3: Compute v_{W}
    v_W = R_W_F16 @ v_F16
    assert v_W.shape == (3,)

    # 4: Back out cos and sin of gamma and sigma.
    cos_sigma = v_W[0]
    sin_sigma = v_W[1]
    sin_gamma = v_W[2]

    out = jnp.array([cos_sigma, sin_sigma, sin_gamma])
    assert out.shape == (3,)
    return out


class F16GCASFloorCeil(Task):
    NX = F16Safeguarded.NX
    NU = F16Safeguarded.NU

    def __init__(self):
        self.dt = 0.05
        self._f16 = F16Safeguarded()

        self.goal_h_min = 50.0
        self.goal_h_max = 150.0
        self.goal_h_width = self.goal_h_max - self.goal_h_min
        self.goal_h_mid = (self.goal_h_min + self.goal_h_max) / 2.0
        self.goal_h_halfwidth = self.goal_h_width / 2

        self.pos_thresh = 200.0

    @override
    def step(self, state: State, control: Control) -> State:
        control = self.discr_to_cts(control)
        self.chk_x(state)
        self.chk_u(control)

        # Integrate using RK4.
        xdot = lambda x: self._f16.xdot(x, control)
        state_new = ode4(xdot, self.dt, state)

        # Safeguard.
        is_valid = self._f16.is_state_valid(state_new)
        freeze_states = ~is_valid
        state_new = state_new.at[F16Safeguarded.FREEZE].set(freeze_states)

        return state_new

    def discr_to_cts(self, control_idx) -> Control:
        """Convert discrete control to continuous."""
        return jnp.array(self.discrete_actions)[control_idx]

    def control_lims(self):
        # [ Nz, Ps, Ny+R ]
        min_u = np.array([-10.0, -10.0, -10.0, 0.0])
        max_u = np.array([15.0, 10.0, 10.0, 1.0])
        return min_u, max_u

    @property
    def discrete_actions(self):
        lb, ub = self.control_lims()
        controls = [
            [lb[0], ub[0]],
            [lb[1], ub[1]],
            [lb[2], ub[2]],
            [lb[3], ub[3]],
        ]
        controls = discretize_u(controls)
        # Add in the zero control.
        controls = np.concatenate([controls, np.zeros((1, 4))], axis=0)
        assert controls.shape == (2**F16Safeguarded.NU + 1, F16Safeguarded.NU)
        return controls

    @property
    def n_actions(self) -> int:
        return len(self.discrete_actions)

    @override
    def get_obs(self, state: State) -> Obs:
        """Encode angles."""
        self.chk_x(state)

        # Learn position-invariant policy.
        state = state.at[F16Safeguarded.PN].set(0.0)

        # sin-cos encode angles.
        with jax.ensure_compile_time_eval():
            angle_idxs = np.array([F16.ALPHA, F16.BETA, F16.PHI, F16.THETA, F16.PSI])
            other_idxs = np.setdiff1d(np.arange(self.NX), angle_idxs)

        angles = state[angle_idxs]
        other = state[other_idxs]

        angles_enc = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=0)
        state_enc = jnp.concatenate([other, angles_enc], axis=0)
        assert state_enc.shape == (self.NX + len(angle_idxs),)

        # Add extra features.
        vel_feats = compute_f16_vel_angles(state[: F16.NX])
        assert vel_feats.shape == (3,)
        state_enc = jnp.concatenate([state_enc, vel_feats], axis=0)

        # fmt: off
        obs_mean = np.array([3.4e+02, -1.7e-01, 2.9e-01, 1.0e-01, 0.0e+00, 1.0e-01, 3.1e+02, 12.0e+00, 3.0e-02, 2.1e-02, 2.4e00, 4.0e-01, 8.7e-01, 8.9e-01, 7.7e-01, 8.0e-01, 7.6e-01, 3.6e-01, -1.3e-02, -7.6e-03, -3.5e-01, -1.4e-03, 5.9e-01, -3.6e-03, 5.4e-01])
        obs_std = np.array([1.1e+02, 1.7e+00, 6.3e-01, 3.2e+00, 1.0e+00, 1.3e+02, 2.2e+02, 5.0e+00, 1.9e+00, 1.5e+00, 4.5e+00, 4.9e-01, 1.4e-01, 9.6e-02, 2.6e-01, 2.2e-01, 3.7e-01, 3.1e-01, 4.4e-01, 5.8e-01, 4.4e-01, 5.4e-01, 3.5e-01, 3.1e-01, 3.8e-01])
        # fmt: on

        state_enc = (state_enc - obs_mean) / obs_std

        # For better stability, clip the state_enc to not be too large.
        state_enc = jnp.clip(state_enc, -10.0, 10.0)

        return state_enc

    @staticmethod
    def train_bounds():
        _MAX_ALT_SAMPLE = 700.0
        # (nx = 17, 2)
        bounds = np.array(
            [
                (150.0, 550.0),  # vt
                MORELLI_BOUNDS[1],  # alpha
                MORELLI_BOUNDS[2],  # beta
                (-np.pi / 3, np.pi / 3),  # phi roll
                (-1.4, 0.4),  # theta pitch
                (-1e-4, 1e-4),  # psi yaw
                (-0.5, 0.5),  # P
                (-0.5, 0.5),  # Q
                (-2 * np.pi, 2 * np.pi),  # R
                (-1000.0, 1000.0),  # pos_n
                (-210.0, 210.0),  # pos_e
                (-10.0, _MAX_ALT_SAMPLE),  # alt.
                (0.0, 10.0),  # power. Consider sampling wider range.
                (-2.0, 2.0),  # nz_int
                (-2.0, 2.0),  # ps_int
                (-2.0, 2.0),  # nyr_int
                (0.0, 0.0),  # is_frozen
            ]
        )
        return bounds

    def sample_x0_train(self, key: PRNGKey, num: int) -> TaskState:
        bounds = self.train_bounds()
        return jr.uniform(key, shape=(num, self.nx), minval=bounds[:, 0], maxval=bounds[:, 1])

    def nominal_val_state(self):
        vt = 5.02089669e02
        alpha = 2.61709746e-02
        pitch = 2.61709746e-02
        alt = 500.0
        power = 7.60335468e00
        Nz_int = 2.43004861e-02
        is_frozen = 0
        # (not exactly) steady state for flying at 500 ft.
        x = np.array([vt, alpha, 0.0, 0.0, pitch, 0, 0.0, 0.0, 0.0, 0.0, 0.0, alt, power, Nz_int, 0, 0, is_frozen])
        return x

    def grid_contour(self) -> tuple[BBFloat, BBFloat, TaskState]:
        # Contour with ( x axis=θ, y axis=H )
        n_xs = 64
        n_ys = 64
        b_th = np.linspace(-1.5, 1.5, num=n_xs)
        b_h = np.linspace(-50.0, 1100.0, num=n_ys)

        x0 = jnp.array(self.nominal_val_state())
        bb_x0 = ei.repeat(x0, "nx -> b1 b2 nx", b1=n_ys, b2=n_xs)

        bb_X, bb_Y = np.meshgrid(b_th, b_h)
        bb_x0 = bb_x0.at[:, :, F16Safeguarded.THETA].set(bb_X)
        bb_x0 = bb_x0.at[:, :, F16Safeguarded.H].set(bb_Y)

        return bb_X, bb_Y, bb_x0

    def get_x0_eval(self, num: int = 128) -> TaskState:
        return self.get_x0_eval_grid(num)
        # with jax.ensure_compile_time_eval():
        #     key = jr.PRNGKey(1234567)
        #     b_x0 = self.sample_x0_train(key, num)
        #     bh_h = jax.vmap(self.h_components)(b_x0)
        #     b_h = jnp.max(bh_h, axis=1)
        #
        #     # Get the 50 most safe ones.
        #     b_idxs = jnp.argsort(b_h)[:50]
        #     b_x0 = b_x0[b_idxs]
        # return b_x0

    def get_x0_eval_grid(self, n_pts: int):
        with jax.ensure_compile_time_eval():
            bb_X, bb_Y, bb_x0 = self.grid_contour()
            b_x0 = merge01(bb_x0)
            bh_h = jax.vmap(self.h_components)(b_x0)
            b_h = jnp.max(bh_h, axis=1)
            b_x0 = b_x0[b_h < -0.2]
            rng = np.random.default_rng(seed=148213)
            idxs = rng.choice(len(b_x0), n_pts, replace=False)
            b_x0 = b_x0[idxs]
        return b_x0

    def should_reset(self, state: State) -> BoolScalar:
        # Reset the state if it is frozen.
        return state[F16Safeguarded.FREEZE] > 0

    @property
    def x_labels(self) -> list[str]:
        return [
            r"$V_t$",
            r"$\alpha$",
            r"$\beta$",
            r"$\phi$",
            r"$\theta$",
            r"$\psi$",
            r"$P$",
            r"$Q$",
            r"$R$",
            r"$Pn$",
            r"$Pe$",
            r"alt",
            r"pow",
            r"$Nz$",
            r"$Ps$",
            r"$Ny+R$",
            r"frozen",
        ]

    @property
    def u_labels(self) -> list[str]:
        return [r"$Nz$", r"$Ps$", r"$Ny+R$", r"thl"]

    @property
    def l_labels(self) -> list[str]:
        return ["cost"]

    @property
    def h_labels(self) -> list[str]:
        return ["alt", r"$\alpha$", r"$\beta$", r"$\theta$", "pos"]
        # return ["alt", r"$\alpha$", r"$\beta$", r"$\theta$", "pos", "freeze"]

    def l_components(self, state: State, control: Control) -> LFloat:
        """Reach target altitude."""
        # Target altitude to be in [h_min=50, h_max=150].
        dist = jnp.clip(jnp.abs(state[F16.H] - self.goal_h_mid) - self.goal_h_halfwidth, a_min=0.0) / 250.0
        # Saturate it.
        cost = jnp.tanh(dist)
        return jnp.array([cost])

    def l(self, state: State, control: Control) -> LFloat:
        weights = jnp.array([1.2e-2])
        l_components = self.l_components(state, control)
        return jnp.sum(l_components * weights)

    def h_altitude(self, state: State) -> FloatScalar:
        """Negative is safe."""
        return 0.3 * box_constr_clipmax(state[F16.H], (0.0, 1000.0), 100.0, 2.0)

    def h_alpha(self, state: State) -> FloatScalar:
        """Negative is safe."""
        return 0.4 * box_constr_log1p(state[F16.ALPHA], MORELLI_BOUNDS[F16.ALPHA], 0.4, -8.0)

    def h_beta(self, state: State) -> FloatScalar:
        """Negative is safe."""
        return 0.4 * box_constr_log1p(state[F16.BETA], MORELLI_BOUNDS[F16.BETA], 0.5, -8.0)

    def h_pitch(self, state: State) -> FloatScalar:
        """Negative is safe."""
        halfpi = 0.95 * np.pi / 2
        return 0.4 * box_constr_log1p(state[F16.THETA], (-halfpi, halfpi), 1.0, -8.0)

    def h_position(self, state: State) -> FloatScalar:
        # Stay within a corridor of [-200, 200] feet along the East/West direction.
        bounds = (-self.pos_thresh, self.pos_thresh)
        return 0.4 * box_constr_log1p(state[F16.PE], bounds, 50.0, -8.0, 10.0, 2.0)

    def h_freeze(self, is_frozen: BoolScalar):
        return jnp.where(is_frozen, jnp.array([8.0]), jnp.array([-8.0]))

    def h_components(self, state: State) -> HFloat:
        """Hard constraints on position and velocity. Negative is safe."""
        self.chk_x(state)

        h_altitude = jnp.atleast_1d(self.h_altitude(state)).max(keepdims=True)
        h_alpha = jnp.atleast_1d(self.h_alpha(state)).max(keepdims=True)
        h_beta = jnp.atleast_1d(self.h_beta(state)).max(keepdims=True)
        h_pitch = jnp.atleast_1d(self.h_pitch(state)).max(keepdims=True)
        h_position = jnp.atleast_1d(self.h_position(state)).max(keepdims=True)

        # is_frozen = state[F16Safeguarded.FREEZE] > 0
        # h_freeze = jnp.atleast_1d(self.h_freeze(is_frozen)).max(keepdims=True)

        # h_h = jnp.concatenate([h_altitude, h_alpha, h_beta, h_pitch, h_position, h_freeze], axis=0)
        h_h = jnp.concatenate([h_altitude, h_alpha, h_beta, h_pitch, h_position], axis=0)
        assert len(h_h) == len(self.h_labels)
        h_h = jnp.where(h_h >= 0, h_h + 0.5, h_h - 0.5)
        return h_h

    def get2d_idxs(self):
        return [F16Safeguarded.THETA, F16Safeguarded.H]

    def setup_traj_plot(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = xlim = [-np.pi / 2, np.pi / 2]
        PLOT_YMIN, PLOT_YMAX = ylim = [-50.0, 1100.0]
        ax.set_facecolor("0.98")
        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel=r"$\theta$", ylabel=r"$h$")

        halfpi = 0.99 * np.pi / 2

        # 2: Plot the avoid set
        obs_style = dict(facecolor="0.6", edgecolor="none", alpha=0.4, zorder=3, hatch="/")
        plot_x_bounds(ax, (-halfpi, halfpi), obs_style)
        plot_y_bounds(ax, (0.0, 1000.0), obs_style)

        # 3: Plot the goal set.
        goal_style = dict(facecolor="green", edgecolor="none", alpha=0.3, zorder=4.0)
        plot_y_goal(ax, (self.goal_h_min, self.goal_h_max), goal_style)

    def setup_traj2_plot(self, axes: list[plt.Axes]):
        # Plot the avoid set.
        obs_style = dict(facecolor="0.6", edgecolor="none", alpha=0.4, zorder=3, hatch="/")
        goal_style = dict(facecolor="green", edgecolor="none", alpha=0.3, zorder=3)
        halfpi = 0.99 * np.pi / 2

        plot_y_bounds(axes[F16Safeguarded.H], (0.0, 1_000.0), obs_style)
        plot_y_bounds(axes[F16Safeguarded.ALPHA], MORELLI_BOUNDS[F16.ALPHA], obs_style)
        plot_y_bounds(axes[F16Safeguarded.BETA], MORELLI_BOUNDS[F16.BETA], obs_style)
        plot_y_bounds(axes[F16Safeguarded.THETA], (-halfpi, halfpi), obs_style)
        plot_y_bounds(axes[F16Safeguarded.PE], (-self.pos_thresh, self.pos_thresh), obs_style)

        # Plot the goal.
        plot_y_goal(axes[F16Safeguarded.H], (self.goal_h_min, self.goal_h_max), goal_style)
