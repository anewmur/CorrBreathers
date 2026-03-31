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


def evaluate_breather_candidate(
    config: dict,
    time_grid: np.ndarray,
    lags: np.ndarray,
    kappa_history: np.ndarray,
    width_history: np.ndarray,
    central_amplitude_history: np.ndarray,
    localization_history: np.ndarray,
) -> dict:
    """Apply simple threshold-based detector for a correlation breather candidate."""
    late_indices = get_late_time_indices(time_grid, config)
    late_width = width_history[late_indices]
    late_central = central_amplitude_history[late_indices]

    initial_central = float(max(abs(central_amplitude_history[0]), 1.0e-14))
    minimum_late_central = float(np.min(np.abs(late_central)))
    central_ratio = minimum_late_central / initial_central

    kappa_zero_series = kappa_history[:, np.where(lags == 0)[0][0]]
    detected_frequency, dominant_peak_ratio = analyze_periodicity(time_grid, kappa_zero_series, late_indices)

    late_average_profile = np.mean(kappa_history[late_indices], axis=0)
    tail_fit_result = fit_tail_models(lags, late_average_profile, config)

    maximum_late_width = float(np.max(late_width))
    average_localization = float(np.mean(localization_history[late_indices]))

    detector_settings = config["detector"]
    localization_ok = maximum_late_width <= float(detector_settings["max_width"])
    longevity_ok = central_ratio >= float(detector_settings["min_central_amplitude_ratio"])
    periodicity_ok = dominant_peak_ratio >= float(detector_settings["min_peak_to_background_ratio"])
    tail_ok = tail_fit_result.error <= float(detector_settings["max_tail_fit_error"])

    found = bool(localization_ok and longevity_ok and periodicity_ok and tail_ok)
    notes = build_notes(localization_ok, longevity_ok, periodicity_ok, tail_ok, tail_fit_result)

    return {
        "found": found,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detected_frequency": float(detected_frequency),
        "central_amplitude": float(central_amplitude_history[-1]),
        "final_width": float(width_history[-1]),
        "oscillatory_tail_detected": bool(tail_fit_result.oscillatory),
        "fitted_alpha": float(tail_fit_result.fitted_alpha),
        "fitted_amplitude": float(tail_fit_result.fitted_amplitude),
        "dominant_peak_ratio": float(dominant_peak_ratio),
        "localization_score": float(average_localization),
        "periodicity_score": float(dominant_peak_ratio),
        "tail_fit_error": float(tail_fit_result.error),
        "central_ratio": float(central_ratio),
        "max_late_width": float(maximum_late_width),
        "notes": notes,
    }


def get_late_time_indices(time_grid: np.ndarray, config: dict) -> np.ndarray:
    """Return index mask for late-time window used by detector."""
    late_time_fraction = float(config["detector"]["late_time_fraction"])
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
    """Estimate dominant frequency and peak-to-background ratio in late window."""
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


def fit_tail_models(lags: np.ndarray, profile: np.ndarray, config: dict) -> TailFitResult:
    """Fit exponential and staggered-exponential tails and keep best model."""
    simple_fit = fit_single_tail_model(lags, profile, oscillatory=False)
    oscillatory_fit = fit_single_tail_model(lags, profile, oscillatory=True)

    prefer_oscillatory = bool(config["detector"].get("oscillatory_tail_preferred", True))
    if prefer_oscillatory and oscillatory_fit.error <= simple_fit.error:
        return oscillatory_fit
    return simple_fit if simple_fit.error <= oscillatory_fit.error else oscillatory_fit


def fit_single_tail_model(lags: np.ndarray, profile: np.ndarray, oscillatory: bool) -> TailFitResult:
    """Least-squares fit for log-amplitude against |k| for one tail model."""
    nonzero_mask = lags != 0
    selected_lags = lags[nonzero_mask]
    selected_profile = profile[nonzero_mask]

    if oscillatory:
        adjusted_profile = selected_profile * ((-1.0) ** np.abs(selected_lags))
        model_name = "oscillatory"
    else:
        adjusted_profile = selected_profile.copy()
        model_name = "monotonic"

    positive_mask = adjusted_profile > 1.0e-12
    fit_lags = np.abs(selected_lags[positive_mask]).astype(float)
    fit_values = adjusted_profile[positive_mask]

    if fit_values.size < 3:
        return TailFitResult(model_name, 1.0e6, 0.0, 0.0, oscillatory)

    design_matrix = np.vstack([np.ones_like(fit_lags), -fit_lags]).T
    target_vector = np.log(fit_values)
    fit_parameters, _, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)

    log_amplitude = float(fit_parameters[0])
    fitted_alpha = float(max(fit_parameters[1], 0.0))
    fitted_amplitude = float(np.exp(log_amplitude))

    predicted = fitted_amplitude * np.exp(-fitted_alpha * fit_lags)
    relative_error = float(np.sqrt(np.mean((predicted - fit_values) ** 2)) / (np.mean(fit_values) + 1.0e-14))

    return TailFitResult(model_name, relative_error, fitted_alpha, fitted_amplitude, oscillatory)


def build_notes(
    localization_ok: bool,
    longevity_ok: bool,
    periodicity_ok: bool,
    tail_ok: bool,
    tail_fit_result: TailFitResult,
) -> str:
    """Build concise detector status message."""
    return (
        f"localization={localization_ok}; "
        f"longevity={longevity_ok}; "
        f"periodicity={periodicity_ok}; "
        f"tail={tail_ok}; "
        f"tail_model={tail_fit_result.model_name}"
    )
