import einops as ei
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Float

from efppo.utils.jax_types import Arr, TFloat, Tp1Float, FloatScalar, BFloat
from efppo.utils.shape_utils import assert_shape


def compute_efocp_gae(
    T_hs: Float[Arr, "T nh"],
    T_l: TFloat,
    T_z: TFloat,
    Tp1_Vh: Float[Arr, "Tp1 nh"],
    Tp1_Vl: Tp1Float,
    disc_gamma: float,
    gae_lambda: float,
    J_max: float,
) -> tuple[Float[Arr, "T nh"], TFloat, TFloat]:
    """Compute GAE for stabilize-avoid. Compute it using DP, starting at V(x_T) and working backwards."""
    T, nh = T_hs.shape

    def loop(carry, inp):
        ii, hs, l, z, Vhs, Vl = inp
        next_Vhs_row, next_Vl_row, gae_coeffs = carry

        mask = assert_shape(jnp.arange(T + 1) < ii + 1, T + 1)
        mask_h = assert_shape(mask[:, None], (T + 1, 1))

        # DP for Vh.
        disc_to_h = (1 - disc_gamma) * hs + disc_gamma * next_Vhs_row
        Vhs_row = assert_shape(mask_h * jnp.maximum(hs, disc_to_h), (T + 1, nh), "Vhs_row")
        # DP for Vl. Clamp it to within J_max so it doesn't get out of hand.
        Vl_row = assert_shape(mask * jnp.minimum(l + disc_gamma * next_Vl_row, J_max), T + 1)

        masked_z = mask * z
        V_row = assert_shape(jnp.maximum(jnp.max(Vhs_row, axis=1), Vl_row - masked_z), (T + 1,))
        cat_V_row = assert_shape(jnp.concatenate([Vhs_row, Vl_row[:, None], V_row[:, None]], axis=1), (T + 1, nh + 2))
        normed_gae_coeffs = assert_shape(gae_coeffs / gae_coeffs.sum(), (T + 1,))
        Qs_GAE = assert_shape(ei.einsum(cat_V_row, normed_gae_coeffs, "Tp1 nhp2, Tp1 -> nhp2"), nh + 2)

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vl_row = Vl_row.at[ii + 1].set(Vl)

        # Update GAE coeffs. [1] -> [λ 1] -> [λ² λ 1]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_coeffs[1] * gae_lambda)

        return (Vhs_row, Vl_row, gae_coeffs), Qs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    T_Vh, T_Vl = Tp1_Vh[:-1], Tp1_Vl[:-1]
    Vh_final, Vl_final = T_Vh[-1], T_Vl[-1]

    init_Vhs = jnp.zeros((T + 1, nh)).at[0, :].set(Vh_final)
    init_Vl = jnp.zeros(T + 1).at[0].set(Vl_final)
    init_carry = (init_Vhs, init_Vl, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, T_hs, T_l, T_z, T_Vh, T_Vl)

    _, Qs_GAEs = lax.scan(loop, init_carry, inps, reverse=True)
    Qhs_GAEs, Ql_GAEs, Q_GAEs = Qs_GAEs[:, :nh], Qs_GAEs[:, nh], Qs_GAEs[:, nh + 1]
    return assert_shape(Qhs_GAEs, (T, nh)), assert_shape(Ql_GAEs, T), assert_shape(Q_GAEs, T)


def compute_efocp_V(z: FloatScalar, Vhs: BFloat, Vl: FloatScalar) -> FloatScalar:
    assert z.shape == Vl.shape, f"z shape {z.shape} should be same as Vl shape {Vl.shape}"
    return jnp.maximum(Vhs.max(), Vl - z)
