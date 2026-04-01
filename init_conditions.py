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

    velocity_ensemble = velocity_ensemble - np.mean(velocity_ensemble, axis=1, keepdims=True)
    return displacement_ensemble, velocity_ensemble


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
