from __future__ import annotations

import numpy as np

from model import compute_strain


def compute_correlation_profiles(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
    spatial_averaging: bool = True,
    window_center_index: int | None = None,
    window_half_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет профили xi_k и kappa_k.

    spatial_averaging=True  — усреднение по ансамблю и по всем узлам (старое поведение).
    spatial_averaging=False — усреднение только по ансамблю в локальном окне.
        Требует window_center_index и window_half_width.
    """
    if spatial_averaging:
        return _compute_profiles_full_chain(
            displacement_ensemble=displacement_ensemble,
            velocity_ensemble=velocity_ensemble,
            lags=lags,
        )

    if window_center_index is None or window_half_width is None:
        raise ValueError(
            "Для spatial_averaging=False нужны window_center_index и window_half_width."
        )
    return _compute_profiles_local_window(
        displacement_ensemble=displacement_ensemble,
        velocity_ensemble=velocity_ensemble,
        lags=lags,
        window_center_index=window_center_index,
        window_half_width=window_half_width,
    )


def _compute_profiles_full_chain(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Усреднение по ансамблю и по всем узлам."""
    strain_ensemble = compute_strain(displacement_ensemble)

    xi_values: list[float] = []
    kappa_values: list[float] = []

    for lag in lags:
        shifted_strain = np.roll(strain_ensemble, shift=-int(lag), axis=1)
        shifted_velocity = np.roll(velocity_ensemble, shift=-int(lag), axis=1)

        xi_values.append(float(np.mean(strain_ensemble * shifted_strain)))
        kappa_values.append(float(np.mean(velocity_ensemble * shifted_velocity)))

    return np.asarray(xi_values, dtype=float), np.asarray(kappa_values, dtype=float)


def _compute_profiles_local_window(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
    window_center_index: int,
    window_half_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Усреднение только по ансамблю в локальном окне."""
    strain_ensemble = compute_strain(displacement_ensemble)
    chain_length = displacement_ensemble.shape[1]

    window_indices = _build_periodic_window_indices(
        chain_length=chain_length,
        window_center_index=window_center_index,
        window_half_width=window_half_width,
    )

    local_strain = strain_ensemble[:, window_indices]
    local_velocity = velocity_ensemble[:, window_indices]

    xi_values: list[float] = []
    kappa_values: list[float] = []

    for lag in lags:
        shifted_strain = np.roll(strain_ensemble, shift=-int(lag), axis=1)[:, window_indices]
        shifted_velocity = np.roll(velocity_ensemble, shift=-int(lag), axis=1)[:, window_indices]

        xi_values.append(float(np.mean(local_strain * shifted_strain)))
        kappa_values.append(float(np.mean(local_velocity * shifted_velocity)))

    return np.asarray(xi_values, dtype=float), np.asarray(kappa_values, dtype=float)


def _build_periodic_window_indices(
    chain_length: int,
    window_center_index: int,
    window_half_width: int,
) -> np.ndarray:
    """Строит индексы локального окна на периодической цепочке."""
    offsets = np.arange(-window_half_width, window_half_width + 1, dtype=int)
    return (int(window_center_index) + offsets) % int(chain_length)


def compute_correlation_width(kappa_profile: np.ndarray, lags: np.ndarray) -> float:
    """Оценивает ширину профиля по второму моменту."""
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
    """Считает долю профиля внутри заданного радиуса локализации."""
    absolute_profile = np.abs(kappa_profile)
    total_weight = float(np.sum(absolute_profile))
    if total_weight <= 1.0e-14:
        return 0.0

    local_weight = float(np.sum(absolute_profile[np.abs(lags) <= int(localization_radius)]))
    return local_weight / total_weight
