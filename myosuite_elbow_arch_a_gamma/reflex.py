"""γ-modulated spinal reflex.

Three modulation mechanisms vs. Architecture A reflex:

1. γs lowers the stretch threshold (intrafusal contraction analogue).
2. γd raises the Ia velocity-sensitivity gain.
3. Reciprocal inhibition picks up the antagonist's γd (the antagonist's
   spindles are also γd-modulated).

Ib (GTO) is unaffected — γ-motors do not innervate tendon organs.
Group II (static stretch) is also left unmodulated for simplicity.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_reflex_gamma(
    L: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    L_opt: np.ndarray,
    F_max: np.ndarray,
    flexors: np.ndarray,
    extensors: np.ndarray,
    gamma_s: np.ndarray,
    gamma_d: np.ndarray,
    G_Ia: float,
    G_Ib: float,
    G_rec: float,
    L_th_ratio: float,
    F_half_ratio: float,
    k_th: float,
    k_v: float,
    Ia_soft_step: bool = False,
    G_II: float = 0.0,
    II_deadband_ratio: float = 0.0,
) -> np.ndarray:
    L_opt_safe = np.maximum(L_opt, 1e-6)
    F_half = np.maximum(F_half_ratio * F_max, 1e-6)

    # γs lowers L_th
    L_th_eff = L_th_ratio * L_opt * (1.0 - k_th * gamma_s)
    if Ia_soft_step:
        Ia_gate = _sigmoid(50.0 * (L - L_th_eff) / L_opt_safe)
    else:
        Ia_gate = (L > L_th_eff).astype(np.float64)

    # γd raises Ia gain (per-muscle)
    gain_Ia_eff = G_Ia * (1.0 + k_v * gamma_d)
    Ia_term = gain_Ia_eff * np.maximum(0.0, V) * Ia_gate

    rel_stretch = (L - L_opt) / L_opt_safe
    II_term = G_II * np.maximum(0.0, rel_stretch - II_deadband_ratio)

    Ib_term = -G_Ib * np.tanh(np.abs(F) / F_half)

    # Reciprocal: antagonist's γd modulates the inhibition each muscle receives
    rec_term = np.zeros_like(L)
    flx_pos_v = np.maximum(0.0, V[flexors])
    ext_pos_v = np.maximum(0.0, V[extensors])
    flx_gate = Ia_gate[flexors]
    ext_gate = Ia_gate[extensors]
    flx_gain = 1.0 + k_v * gamma_d[flexors]
    ext_gain = 1.0 + k_v * gamma_d[extensors]

    flx_drive = float(np.sum(flx_gain * flx_pos_v * flx_gate))
    ext_drive = float(np.sum(ext_gain * ext_pos_v * ext_gate))
    rec_term[flexors] = -G_rec * ext_drive
    rec_term[extensors] = -G_rec * flx_drive

    return Ia_term + II_term + Ib_term + rec_term
