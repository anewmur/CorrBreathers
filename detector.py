from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass
class TailFitResult:
    model_name: str
    error: float
    fitted_alpha: float
    fitted_amplitude: float
    oscillatory: bool
    sign_mismatch_fraction: float


def evaluate_breather_candidate(
    detector_settings: dict,
    time_grid: np.ndarray,
    lags: np.ndarray,
    kappa_history: np.ndarray,
    width_history: np.ndarray,
    central_amplitude_history: np.ndarray,
    localization_history: np.ndarray,
) -> dict:
    """Проверяет, проходит ли профиль пороги детектора."""
    late_indices = get_late_time_indices(time_grid, detector_settings)
    late_width = width_history[late_indices]
    late_central = central_amplitude_history[late_indices]

    initial_central = float(max(abs(central_amplitude_history[0]), 1.0e-14))
    minimum_late_central = float(np.min(np.abs(late_central)))
    central_ratio = minimum_late_central / initial_central

    kappa_zero_index = int(np.where(lags == 0)[0][0])
    kappa_zero_series = kappa_history[:, kappa_zero_index]
    detected_frequency, dominant_peak_ratio = analyze_periodicity(time_grid, kappa_zero_series, late_indices)

    late_average_profile = np.mean(kappa_history[late_indices], axis=0)
    minimum_late_spectrum_value, passes_psd_check = check_profile_toeplitz_psd(
        lags,
        late_average_profile,
        float(detector_settings.get("profile_psd_tolerance", 1.0e-10)),
    )
    tail_fit_result = fit_tail_models(
        lags,
        late_average_profile,
        int(detector_settings.get("min_tail_fit_lag", 4)),
        bool(detector_settings.get("oscillatory_tail_preferred", True)),
    )

    maximum_late_width = float(np.max(late_width))
    average_localization = float(np.mean(localization_history[late_indices]))

    localization_ok = maximum_late_width <= float(detector_settings["max_width"])
    longevity_ok = central_ratio >= float(detector_settings["min_central_amplitude_ratio"])
    periodicity_ok = dominant_peak_ratio >= float(detector_settings["min_peak_to_background_ratio"])
    tail_ok = tail_fit_result.error <= float(detector_settings["max_tail_fit_error"])
    psd_ok = bool(passes_psd_check)

    found = bool(localization_ok and longevity_ok and periodicity_ok and tail_ok and psd_ok)
    notes = build_notes(
        localization_ok,
        longevity_ok,
        periodicity_ok,
        tail_ok,
        psd_ok,
        tail_fit_result,
        minimum_late_spectrum_value,
    )

    return {
        "found": found,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detected_frequency": float(detected_frequency),
        "central_amplitude": float(central_amplitude_history[-1]),
        "final_width": float(width_history[-1]),
        "oscillatory_tail_detected": bool(tail_fit_result.oscillatory),
        "tail_model": str(tail_fit_result.model_name),
        "fitted_alpha": float(tail_fit_result.fitted_alpha),
        "fitted_amplitude": float(tail_fit_result.fitted_amplitude),
        "dominant_peak_ratio": float(dominant_peak_ratio),
        "localization_score": float(average_localization),
        "periodicity_score": float(dominant_peak_ratio),
        "tail_fit_error": float(tail_fit_result.error),
        "sign_mismatch_fraction": float(tail_fit_result.sign_mismatch_fraction),
        "central_ratio": float(central_ratio),
        "max_late_width": float(maximum_late_width),
        "minimum_late_spectrum_value": float(minimum_late_spectrum_value),
        "passes_psd_check": bool(passes_psd_check),
        "notes": notes,
    }


def get_late_time_indices(time_grid: np.ndarray, detector_settings: dict) -> np.ndarray:
    """Возвращает индексы позднего временного окна."""
    late_time_fraction = float(detector_settings["late_time_fraction"])
    threshold_time = late_time_fraction * float(time_grid[-1])
    late_indices = np.where(time_grid >= threshold_time)[0]
    if late_indices.size == 0:
        return np.arange(time_grid.size)
    return late_indices


def analyze_periodicity(
    time_grid: np.ndarray,
    kappa_zero_series: np.ndarray,
    late_indices: np.ndarray,
) -> tuple[float, float]:
    """Оценивает главную частоту и контраст пика спектра."""
    late_signal = kappa_zero_series[late_indices]
    detrended_signal = late_signal - np.mean(late_signal)

    if detrended_signal.size < 4:
        return 0.0, 0.0

    time_step = float(np.mean(np.diff(time_grid[late_indices])))
    frequency_grid = np.fft.rfftfreq(detrended_signal.size, d=time_step)
    amplitude_spectrum = np.abs(np.fft.rfft(detrended_signal))

    if amplitude_spectrum.size <= 1:
        return 0.0, 0.0

    nonzero_spectrum = amplitude_spectrum[1:]
    nonzero_frequencies = frequency_grid[1:]
    peak_index = int(np.argmax(nonzero_spectrum))
    dominant_amplitude = float(nonzero_spectrum[peak_index])
    detected_frequency = float(nonzero_frequencies[peak_index])

    median_background = float(np.median(nonzero_spectrum) + 1.0e-14)
    dominant_peak_ratio = dominant_amplitude / median_background
    return detected_frequency, dominant_peak_ratio


def check_profile_toeplitz_psd(
    lags: np.ndarray,
    profile: np.ndarray,
    psd_tolerance: float,
) -> tuple[float, bool]:
    """Проверяет допустимость профиля по минимальному собственному числу Тёплица."""
    symmetric_profile = 0.5 * (profile + profile[::-1])
    nonnegative_lag_mask = lags >= 0
    nonnegative_lags = lags[nonnegative_lag_mask]
    nonnegative_profile = symmetric_profile[nonnegative_lag_mask]
    sorted_indices = np.argsort(nonnegative_lags)
    correlation_values = nonnegative_profile[sorted_indices]

    matrix_size = correlation_values.size
    toeplitz_matrix = np.empty((matrix_size, matrix_size), dtype=float)
    for row_index in range(matrix_size):
        for column_index in range(matrix_size):
            lag_distance = abs(row_index - column_index)
            toeplitz_matrix[row_index, column_index] = correlation_values[lag_distance]

    eigenvalues = np.linalg.eigvalsh(toeplitz_matrix)
    minimum_value = float(np.min(eigenvalues))
    passes_check = bool(minimum_value >= -psd_tolerance)
    return minimum_value, passes_check


def fit_tail_models(
    lags: np.ndarray,
    profile: np.ndarray,
    min_tail_fit_lag: int,
    oscillatory_tail_preferred: bool,
) -> TailFitResult:
    """Сравнивает две модели хвоста и выбирает лучшую."""
    simple_fit = fit_single_tail_model(lags, profile, min_tail_fit_lag, oscillatory=False)
    oscillatory_fit = fit_single_tail_model(lags, profile, min_tail_fit_lag, oscillatory=True)

    if oscillatory_tail_preferred and oscillatory_fit.error <= simple_fit.error:
        return oscillatory_fit
    return simple_fit if simple_fit.error <= oscillatory_fit.error else oscillatory_fit


def fit_single_tail_model(
    lags: np.ndarray,
    profile: np.ndarray,
    min_tail_fit_lag: int,
    oscillatory: bool,
) -> TailFitResult:
    """Строит экспоненциальный фит для выбранной модели хвоста."""
    outer_mask = np.abs(lags) >= int(min_tail_fit_lag)
    selected_lags = lags[outer_mask]
    selected_profile = profile[outer_mask]

    if oscillatory:
        adjusted_profile = selected_profile * ((-1.0) ** np.abs(selected_lags))
        model_name = "oscillatory"
    else:
        adjusted_profile = selected_profile.copy()
        model_name = "monotonic"

    fit_lags = np.abs(selected_lags).astype(float)
    absolute_values = np.maximum(np.abs(adjusted_profile), 1.0e-12)
    sign_mismatch_fraction = float(np.mean(adjusted_profile <= 0.0))
    sign_mismatch_penalty = sign_mismatch_fraction

    if absolute_values.size < 3:
        return TailFitResult(model_name, 1.0e6, 0.0, 0.0, oscillatory, sign_mismatch_fraction)

    design_matrix = np.vstack([np.ones_like(fit_lags), -fit_lags]).T
    target_vector = np.log(absolute_values)
    fit_parameters, _, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)

    log_amplitude = float(fit_parameters[0])
    fitted_alpha = float(max(fit_parameters[1], 0.0))
    fitted_amplitude = float(np.exp(log_amplitude))

    predicted = fitted_amplitude * np.exp(-fitted_alpha * fit_lags)
    relative_error = float(
        np.sqrt(np.mean((predicted - adjusted_profile) ** 2)) / (np.mean(absolute_values) + 1.0e-14)
        + sign_mismatch_penalty
    )

    return TailFitResult(
        model_name,
        relative_error,
        fitted_alpha,
        fitted_amplitude,
        oscillatory,
        sign_mismatch_fraction,
    )


def build_notes(
    localization_ok: bool,
    longevity_ok: bool,
    periodicity_ok: bool,
    tail_ok: bool,
    psd_ok: bool,
    tail_fit_result: TailFitResult,
    minimum_late_spectrum_value: float,
) -> str:
    """Собирает короткое пояснение к решению детектора."""
    return (
        f"localization={localization_ok}; "
        f"longevity={longevity_ok}; "
        f"periodicity={periodicity_ok}; "
        f"tail={tail_ok}; "
        f"tail_model={tail_fit_result.model_name}; "
        f"tail_sign_mismatch={tail_fit_result.sign_mismatch_fraction:.3f}; "
        f"psd={psd_ok}; "
        f"min_toeplitz_eigenvalue={minimum_late_spectrum_value:.3e}"
    )
