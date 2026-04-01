from __future__ import annotations

import numpy as np


def create_initial_ensemble(
    ensemble_size: int,
    chain_length: int,
    mode: str,
    zero_displacements: bool,
    random_thermal_std: float,
    amplitude: float,
    alpha: float,
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
    remove_center_of_mass_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт начальные смещения и скорости для ансамбля."""
    if zero_displacements:
        displacement_ensemble = np.zeros((ensemble_size, chain_length), dtype=float)
    else:
        displacement_ensemble = np.random.normal(0.0, 1.0, size=(ensemble_size, chain_length))

    if mode == "random_thermal":
        velocity_ensemble = sample_random_thermal_velocities(
            ensemble_size,
            chain_length,
            random_thermal_std,
        )
    elif mode == "correlated_velocity":
        velocity_ensemble = sample_correlated_velocities_fft(
            ensemble_size,
            chain_length,
            amplitude,
            alpha,
            covariance_psd_tolerance,
            covariance_negative_warning_tolerance,
        )
    else:
        raise ValueError(f"Неизвестный initial_conditions.mode: {mode}")

    if remove_center_of_mass_velocity:
        velocity_ensemble = velocity_ensemble - np.mean(velocity_ensemble, axis=1, keepdims=True)
        print(
            "ПРЕДУПРЕЖДЕНИЕ: вычтена средняя скорость по узлам; "
            "ковариация может не совпадать с заданной."
        )

    return displacement_ensemble, velocity_ensemble


def validate_initial_covariance(
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
    target_profile: np.ndarray,
) -> dict:
    """Сравнивает эмпирический профиль kappa_k(0) с целевым профилем."""
    empirical_values: list[float] = []

    for lag in lags:
        # Совпадает с конвенцией в compute_correlation_profiles: shift = -lag.
        shifted_velocity = np.roll(velocity_ensemble, shift=-int(lag), axis=1)
        empirical_values.append(float(np.mean(velocity_ensemble * shifted_velocity)))

    empirical_profile = np.asarray(empirical_values, dtype=float)
    target_profile_array = np.asarray(target_profile, dtype=float)

    max_abs_error = float(np.max(np.abs(empirical_profile - target_profile_array)))

    target_norm = float(np.linalg.norm(target_profile_array))
    if target_norm <= 1.0e-14:
        relative_error = 0.0
    else:
        relative_error = float(np.linalg.norm(empirical_profile - target_profile_array) / target_norm)

    return {
        "empirical_profile": empirical_profile,
        "target_profile": target_profile_array,
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
    }


def sample_random_thermal_velocities(
    ensemble_size: int,
    chain_length: int,
    thermal_std: float,
) -> np.ndarray:
    """Генерирует независимые гауссовы скорости."""
    return np.random.normal(0.0, thermal_std, size=(ensemble_size, chain_length))


def sample_correlated_velocities_fft(
    ensemble_size: int,
    chain_length: int,
    amplitude: float,
    alpha: float,
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
) -> np.ndarray:
    """Генерирует скорости через БПФ с циркулянтной ковариацией."""
    lag_index = np.arange(chain_length, dtype=float)
    lag_index = np.minimum(lag_index, chain_length - lag_index)
    correlation_values = amplitude * ((-1.0) ** lag_index) * np.exp(-alpha * lag_index)

    clipped_spectrum = validate_and_prepare_spectrum(
        correlation_values,
        covariance_psd_tolerance,
        covariance_negative_warning_tolerance,
    )
    filter_amplitudes = np.sqrt(np.maximum(clipped_spectrum, 0.0))

    white_noise = np.random.normal(0.0, 1.0, size=(ensemble_size, chain_length))
    white_noise_spectrum = np.fft.fft(white_noise, axis=1)
    shaped_spectrum = white_noise_spectrum * filter_amplitudes[None, :]
    sampled_velocities = np.real(np.fft.ifft(shaped_spectrum, axis=1))
    return sampled_velocities


def validate_and_prepare_spectrum(
    correlation_values: np.ndarray,
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
) -> np.ndarray:
    """Проверяет спектр плотности и обрезает малые отрицательные значения."""
    spectrum = np.real(np.fft.fft(correlation_values))
    minimum_spectrum_value = float(np.min(spectrum))

    if minimum_spectrum_value >= -covariance_psd_tolerance:
        return np.clip(spectrum, a_min=0.0, a_max=None)

    if minimum_spectrum_value >= -covariance_negative_warning_tolerance:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: в спектре ковариации есть малые отрицательные значения; "
            "они обрезаны до нуля."
        )
        return np.clip(spectrum, a_min=0.0, a_max=None)

    raise ValueError(
        "Спектральная плотность ковариации заметно отрицательна. "
        f"Минимум спектра: {minimum_spectrum_value:.3e}."
    )
