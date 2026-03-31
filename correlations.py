from __future__ import annotations

import numpy as np

from model import compute_strain


def compute_correlation_profiles(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute xi_k and kappa_k by averaging over sites and ensemble members."""
    strain_ensemble = compute_strain(displacement_ensemble)

    xi_values: list[float] = []
    kappa_values: list[float] = []

    for lag in lags:
        shifted_strain = np.roll(strain_ensemble, shift=-int(lag), axis=1)
        shifted_velocity = np.roll(velocity_ensemble, shift=-int(lag), axis=1)

        xi_value = float(np.mean(strain_ensemble * shifted_strain))
        kappa_value = float(np.mean(velocity_ensemble * shifted_velocity))

        xi_values.append(xi_value)
        kappa_values.append(kappa_value)

    return np.asarray(xi_values, dtype=float), np.asarray(kappa_values, dtype=float)


def compute_correlation_width(kappa_profile: np.ndarray, lags: np.ndarray) -> float:
    """Compute effective width sqrt(sum(k^2 w_k)/sum(w_k)) with positive weights."""
    weights = np.abs(kappa_profile)
    denominator = float(np.sum(weights))
    if denominator <= 1.0e-14:
        return float(np.max(np.abs(lags)))

    second_moment = float(np.sum((lags.astype(float) ** 2) * weights) / denominator)
    return float(np.sqrt(max(second_moment, 0.0)))


def compute_localization_fraction(
    kappa_profile: np.ndarray,
    lags: np.ndarray,
    localization_radius: int,
) -> float:
    """Fraction of |kappa_k| concentrated in |k| <= localization_radius."""
    absolute_profile = np.abs(kappa_profile)
    total_weight = float(np.sum(absolute_profile))
    if total_weight <= 1.0e-14:
        return 0.0

    local_weight = float(np.sum(absolute_profile[np.abs(lags) <= int(localization_radius)]))
    return local_weight / total_weight
