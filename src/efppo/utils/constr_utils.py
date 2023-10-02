from efppo.utils.jax_types import FloatScalar


def box_constraint(x: FloatScalar, lb: FloatScalar, ub: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """Box constraint on x. Return [ -(x - lb), x - ub ]"""
    return -(x - lb), x - ub
