from efppo.utils.jax_types import Arr, Float

State = Float[Arr, "nx"]
Control = Float[Arr, "nu"]
Obs = Float[Arr, "nobs"]
Sample = Float[Arr, "*"]

BState = Float[Arr, "b nx"]
BControl = Float[Arr, "b nu"]
BObs = Float[Arr, "b nobs"]

LFloat = Float[Arr, "nl"]
BLFloat = Float[Arr, "b nl"]
ZBLFloat = Float[Arr, "nz b nl"]

HFloat = Float[Arr, "nh"]
BHFloat = Float[Arr, "b nh"]
ZBHFloat = Float[Arr, "nz b nh"]

BBState = Float[Arr, "b1 b2 nx"]
BBControl = Float[Arr, "b1 b2 nu"]

BBTControl = Float[Arr, "b1 b2 T nu"]

ZFloat = Float[Arr, "nz"]
ZBFloat = Float[Arr, "nz b"]
ZBBFloat = Float[Arr, "nz b1 b2"]
ZBTState = Float[Arr, "nz b T nx"]
ZBBControl = Float[Arr, "nz b1 b2 nu"]

THFloat = Float[Arr, "T nh"]
BTHFloat = Float[Arr, "b T nh"]
ZBTHFloat = Float[Arr, "nz b T nh"]

BBHFloat = Float[Arr, "b1 b2 nh"]
ZBBHFloat = Float[Arr, "nz b1 b2 nh"]
BBTHFloat = Float[Arr, "b1 b2 T nh"]

BTState = Float[Arr, "b T nx"]
BTObs = Float[Arr, "b T nobs"]
BTSample = Float[Arr, "b T *"]

FxShape = Float[Arr, "nx nx"]
FuShape = Float[Arr, "nx nu"]

TState = Float[Arr, "T nx"]
TObs = Float[Arr, "T nobs"]
TControl = Float[Arr, "T nu"]
