import jax.numpy as jnp
import numpy as np
from jax_f16.f16 import F16

from efppo.task.dyn_types import Control, State
from efppo.utils.jax_types import BoolScalar

# The bounds on alpha and beta come from the bounds of the data used for the Morelli model.
MORELLI_BOUNDS = np.array(
    [
        [-np.inf, np.inf],  # vt
        # [_d2r(-10), _d2r(45)],  # alpha (rad)
        [-0.17453292519943295, 0.7853981633974483],  # alpha (rad)
        # [_d2r(-30), _d2r(30)],  # beta (rad)
        [-0.5235987755982988, 0.5235987755982988],  # beta (rad)
        [-np.inf, np.inf],  # roll (rad)
        [-np.inf, np.inf],  # pitch (rad)
        [-np.inf, np.inf],  # yaw (rad)
        [-np.inf, np.inf],  # P
        [-np.inf, np.inf],  # Q
        [-np.inf, np.inf],  # R
        [-np.inf, np.inf],  # north pos
        [-np.inf, np.inf],  # east pos
        [-np.inf, np.inf],  # altitude
        [-np.inf, np.inf],  # engine thrust dynamics lag state
        [-np.inf, np.inf],  # Nz integrator
        [-np.inf, np.inf],  # Ps integrator
        [-np.inf, np.inf],  # Ny+R integrator
    ]
)


def _in_morelli(state: State, idx: int, eps: float) -> BoolScalar:
    return (MORELLI_BOUNDS[idx, 0] - eps <= state[idx]) & (state[idx] <= MORELLI_BOUNDS[idx, 1] + eps)


class F16Safeguarded:
    """Represents a safeguarded version of the F16 aircraft, where the dynamics are stopped when the
    angles go out of bounds to prevent numerical instability. An additional integrator state is added that keeps track
    of this safeguard state, expanding NX from 16 -> 17.

    The system has state
        x[0] = air speed, VT    (ft/sec)
        x[1] = angle of attack, alpha  (rad)
        x[2] = angle of sideslip, beta (rad)
        x[3] = roll angle, phi  (rad)
        x[4] = pitch angle, theta  (rad)
        x[5] = yaw angle, psi  (rad)
        x[6] = roll rate, P  (rad/sec)
        x[7] = pitch rate, Q  (rad/sec)
        x[8] = yaw rate, R  (rad/sec)
        x[9] = northward horizontal displacement, pn  (feet)
        x[10] = eastward horizontal displacement, pe  (feet)
        x[11] = altitude, h  (feet)
        x[12] = engine thrust dynamics lag state, pow
        x[13, 14, 15] = internal integrator states
        x[16] = dynamics currently frozen if >0
    and control inputs, which are setpoints for a lower-level integrator
        u[0] = Z acceleration
        u[1] = stability roll rate
        u[2] = side acceleration + yaw rate (usually regulated to 0)
        u[3] = throttle command (0.0, 1.0)
    """

    NX = F16.NX + 1
    NU = F16.NU

    VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, PN, PE, H, POW, NZINT, PSINT, NYRINT, FREEZE = range(NX)
    NZ, PS, NYR, THRTL = range(NU)

    def __init__(self):
        self._f16 = F16()

    def is_state_valid(self, state: State):
        not_frozen = state[F16Safeguarded.FREEZE] <= 0
        angle_eps = 0.1
        in_morelli = _in_morelli(state, F16.ALPHA, angle_eps) & _in_morelli(state, F16.BETA, angle_eps)
        pitch_valid = jnp.abs(state[F16.THETA]) < np.pi / 2

        alt_valid = (-100 <= state[F16.H]) & (state[F16.H] <= 1_100.0)
        pe_valid = (-250.0 <= state[F16.PE]) & (state[F16.PE] <= 250.0)
        rollvel_valid = jnp.abs(state[F16.P]) < 10.0

        return not_frozen & in_morelli & pitch_valid & alt_valid & pe_valid & rollvel_valid

    def xdot(self, state: State, control: Control) -> State:
        assert state.shape == (F16Safeguarded.NX,)
        assert control.shape == (F16Safeguarded.NU,)

        f16_state = state[: F16.NX]
        f16_state_dot = self._f16.xdot(f16_state, control)
        not_frozen = state[F16Safeguarded.FREEZE] <= 0

        valid_xdot = jnp.concatenate([f16_state_dot, jnp.zeros(1)])
        invalid_xdot = jnp.concatenate([jnp.zeros(F16.NX), jnp.zeros(1)])
        return jnp.where(not_frozen, valid_xdot, invalid_xdot)
