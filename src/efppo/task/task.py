from typing import TypeVar

import matplotlib.pyplot as plt

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State
from efppo.utils.jax_types import AnyFloat, BBFloat
from efppo.utils.rng import PRNGKey
from efppo.utils.shape_utils import assert_shape

TaskState = TypeVar("TaskState")


class Task:
    NX = None
    NU = None
    NOBS = None

    def step(self, state: State, control: Control) -> State:
        ...

    def get_obs(self, state: State) -> Obs:
        ...

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def nx(self) -> int:
        return self.NX

    @property
    def nu(self) -> int:
        return self.NU

    @property
    def n_actions(self) -> int:
        ...

    def sample_x0_train(self, key: PRNGKey, num: int) -> TaskState:
        raise NotImplementedError("")

    def grid_contour(self) -> tuple[BBFloat, BBFloat, TaskState]:
        raise NotImplementedError("")

    def get_x0_eval(self) -> TaskState:
        raise NotImplementedError("")

    @property
    def nobs(self) -> int:
        return self.NOBS

    @property
    def nh(self) -> int:
        return len(self.h_labels)

    @property
    def x_labels(self) -> list[str]:
        raise NotImplementedError("")

    @property
    def u_labels(self) -> list[str]:
        raise NotImplementedError("")

    @property
    def l_labels(self) -> list[str]:
        ...

    @property
    def h_labels(self) -> list[str]:
        ...

    def l(self, state: State, control: Control) -> LFloat:
        ...

    def h_components(self, state: State) -> HFloat:
        ...

    def chk_x(self, state: State) -> State:
        return assert_shape(state, self.nx, "state")

    def chk_u(self, control: Control) -> Control:
        return assert_shape(control, self.nu, "control")

    def chk_obs(self, obs: Obs) -> Obs:
        return assert_shape(obs, self.nobs, "observation")

    def setup_traj_plot(self, ax: plt.Axes):
        ax.set(xlabel="x[0]", ylabel="x[1]")

    def get2d(self, state: State) -> tuple[AnyFloat, AnyFloat]:
        return state[..., 0], state[..., 1]
