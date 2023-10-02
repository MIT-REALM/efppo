from typing import NamedTuple, Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, Shaped

Arr = Union[np.ndarray, Array]

Shape = tuple[int, ...]

AnyFloat = Float[Arr, "*"]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

BFloat = Float[Arr, "b"]
BInt = Int[Arr, "b"]
BBool = Bool[Arr, "b"]

TFloat = Float[Arr, "T"]
Tp1Float = Float[Arr, "Tp1"]

BTFloat = Float[Arr, "b T"]
BTInt = Int[Arr, "b T"]
BTBool = Bool[Arr, "b T"]

BHFloat = Float[Arr, "b h"]
BHInt = Int[Arr, "b h"]
BHBool = Bool[Arr, "b h"]

BLFloat = Float[Arr, "b l"]
BLInt = Int[Arr, "b l"]
BLBool = Bool[Arr, "b l"]

BTHFloat = Float[Arr, "b T h"]
BTHInt = Int[Arr, "b T h"]
BTHBool = Bool[Arr, "b T h"]

BTLFloat = Float[Arr, "b T l"]
BTLInt = Int[Arr, "b T l"]
BTLBool = Bool[Arr, "b T l"]

BBFloat = Float[Arr, "b1 b2"]
BBBool = Bool[Arr, "b1 b2"]
BBInt = Int[Arr, "b1 b2"]

MetricsDict = dict[str, FloatScalar]
