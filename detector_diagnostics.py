from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class TailFitResult:
    model_name: str
    error: float
    fitted_alpha: float
    fitted_amplitude: float
    oscillatory: bool
    sign_mismatch_fraction: float


@dataclass(slots=True)
class SpectrumMetric:
    profile_name: str
    lag: int
    dominant_frequency: float
    dominant_period: float
    dominant_peak_ratio: float


@dataclass(slots=True)
class ProfileDiagnostics:
    profile_name: str
    saved_steps: int
    total_time: float
    lag_min: int
    lag_max: int
    initial_width: float
    final_width: float
    min_late_width: float
    mean_late_width: float
    max_late_width: float
    initial_central_amplitude: float
    final_central_amplitude: float
    min_abs_late_central_amplitude: float
    late_to_initial_central_amplitude_ratio: float
    min_late_localization_fraction: float
    mean_late_localization_fraction: float
    max_late_localization_fraction: float
    dominant_frequency_lag_0: float
    dominant_period_lag_0: float
    dominant_peak_ratio_lag_0: float
    minimum_toeplitz_eigenvalue_on_late_mean_profile: float
    passes_psd_check: bool
    monotonic_tail_alpha: float
    monotonic_tail_error: float
    oscillatory_tail_alpha: float
    oscillatory_tail_error: float
    oscillatory_tail_sign_mismatch_fraction: float
    best_tail_model: str
    best_tail_alpha: float
    best_tail_error: float
    recurrence_shift_steps: int
    recurrence_time_shift: float
    recurrence_mean_relative_distance: float
    recurrence_max_relative_distance: float


@dataclass(slots=True)
class DiagnosticsConfig:
    npz_path: Path
    output_directory: Path
    late_time_fraction: float = 0.5
    min_tail_fit_lag: int = 4
    spectrum_lags: list[int] | None = None
    heatmap_value_quantile: float = 0.995
    tail_profile_count: int = 3


REQUIRED_ARRAY_NAMES = [
    "time_grid",
    "lags",
    "xi_history",
    "kappa_history",
    "xi_width_history",
    "kappa_width_history",
    "xi_central_amplitude_history",
    "kappa_central_amplitude_history",
    "xi_localization_history",
    "kappa_localization_history",
    "energy_history",
    "relative_energy_drift_history",
]


def load_time_series(npz_path: Path) -> dict[str, np.ndarray]:
    """Загружает и валидирует сохранённые массивы расчёта."""
    loaded = np.load(npz_path)
    arrays: dict[str, np.ndarray] = {}

    for array_name in REQUIRED_ARRAY_NAMES:
        if array_name not in loaded:
            raise KeyError(f"В файле {npz_path} отсутствует массив {array_name!r}")
        arrays[array_name] = loaded[array_name]

    return arrays


def build_default_config(
    npz_path: str | Path = "results/time_series.npz",
    output_directory: str | Path | None = None,
    late_time_fraction: float = 0.5,
    min_tail_fit_lag: int = 4,
    spectrum_lags: list[int] | None = None,
    heatmap_value_quantile: float = 0.995,
    tail_profile_count: int = 3,
) -> DiagnosticsConfig:
    """Собирает конфигурацию диагностики с разумными значениями по умолчанию."""
    npz_path_obj = Path(npz_path)
    if output_directory is None:
        output_directory_obj = npz_path_obj.parent / "diagnostics"
    else:
        output_directory_obj = Path(output_directory)

    if spectrum_lags is None:
        spectrum_lags = [0, 1, 2, 3]

    return DiagnosticsConfig(
        npz_path=npz_path_obj,
        output_directory=output_directory_obj,
        late_time_fraction=late_time_fraction,
        min_tail_fit_lag=min_tail_fit_lag,
        spectrum_lags=list(spectrum_lags),
        heatmap_value_quantile=heatmap_value_quantile,
        tail_profile_count=tail_profile_count,
    )


def ensure_output_directory(output_directory: Path) -> None:
    """Создаёт каталог для результатов диагностики."""
    output_directory.mkdir(parents=True, exist_ok=True)


def get_late_time_indices(time_grid: np.ndarray, late_time_fraction: float) -> np.ndarray:
    """Возвращает индексы позднего временного окна."""
    if time_grid.size == 0:
        return np.asarray([], dtype=int)

    threshold_time = late_time_fraction * float(time_grid[-1])
    late_indices = np.where(time_grid >= threshold_time)[0]
    if late_indices.size == 0:
        return np.arange(time_grid.size, dtype=int)
    return late_indices.astype(int)


def get_lag_index(lags: np.ndarray, lag_value: int) -> int:
    """Возвращает индекс заданного лага."""
    matched_indices = np.where(lags == int(lag_value))[0]
    if matched_indices.size == 0:
        raise ValueError(f"Лаг {lag_value} отсутствует в массиве lags")
    return int(matched_indices[0])


def get_available_spectrum_lags(lags: np.ndarray, requested_lags: list[int]) -> list[int]:
    """Оставляет только те лаги, которые реально присутствуют в данных."""
    available_lags: list[int] = []
    for lag_value in requested_lags:
        if int(lag_value) in set(int(value) for value in lags.tolist()):
            available_lags.append(int(lag_value))
    if not available_lags:
        available_lags.append(0)
    return available_lags


def analyze_periodicity(
    time_grid: np.ndarray,
    signal: np.ndarray,
    late_indices: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Оценивает доминирующую частоту, период и контраст спектрального пика."""
    late_signal = signal[late_indices]
    detrended_signal = late_signal - np.mean(late_signal)

    if detrended_signal.size < 4:
        return 0.0, float("nan"), 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)

    late_time_grid = time_grid[late_indices]
    average_time_step = float(np.mean(np.diff(late_time_grid)))
    if average_time_step <= 0.0:
        return 0.0, float("nan"), 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)

    frequency_grid = np.fft.rfftfreq(detrended_signal.size, d=average_time_step)
    amplitude_spectrum = np.abs(np.fft.rfft(detrended_signal))

    if amplitude_spectrum.size <= 1:
        return 0.0, float("nan"), 0.0, frequency_grid, amplitude_spectrum

    positive_frequencies = frequency_grid[1:]
    positive_amplitudes = amplitude_spectrum[1:]
    peak_index = int(np.argmax(positive_amplitudes))
    dominant_frequency = float(positive_frequencies[peak_index])
    dominant_amplitude = float(positive_amplitudes[peak_index])
    background_level = float(np.median(positive_amplitudes) + 1.0e-14)
    dominant_peak_ratio = dominant_amplitude / background_level

    if dominant_frequency > 0.0:
        dominant_period = 1.0 / dominant_frequency
    else:
        dominant_period = float("nan")

    return dominant_frequency, dominant_period, dominant_peak_ratio, frequency_grid, amplitude_spectrum


def check_profile_toeplitz_psd(
    lags: np.ndarray,
    profile: np.ndarray,
    psd_tolerance: float = 1.0e-10,
) -> tuple[float, bool]:
    """Проверяет допустимость корреляционного профиля по матрице Тёплица."""
    symmetric_profile = 0.5 * (profile + profile[::-1])
    nonnegative_mask = lags >= 0
    nonnegative_lags = lags[nonnegative_mask]
    nonnegative_profile = symmetric_profile[nonnegative_mask]

    sorted_indices = np.argsort(nonnegative_lags)
    correlation_values = nonnegative_profile[sorted_indices]
    matrix_size = int(correlation_values.size)

    toeplitz_matrix = np.empty((matrix_size, matrix_size), dtype=float)
    for row_index in range(matrix_size):
        for column_index in range(matrix_size):
            lag_distance = abs(row_index - column_index)
            toeplitz_matrix[row_index, column_index] = correlation_values[lag_distance]

    eigenvalues = np.linalg.eigvalsh(toeplitz_matrix)
    minimum_value = float(np.min(eigenvalues))
    passes_check = bool(minimum_value >= -psd_tolerance)
    return minimum_value, passes_check


def fit_single_tail_model(
    lags: np.ndarray,
    profile: np.ndarray,
    min_tail_fit_lag: int,
    oscillatory: bool,
) -> TailFitResult:
    """Подгоняет экспоненциальный хвост на позднем среднем профиле."""
    outer_mask = np.abs(lags) >= int(min_tail_fit_lag)
    selected_lags = lags[outer_mask]
    selected_profile = profile[outer_mask]

    if oscillatory:
        adjusted_profile = selected_profile * ((-1.0) ** np.abs(selected_lags))
        model_name = "oscillatory"
    else:
        adjusted_profile = selected_profile.copy()
        model_name = "monotonic"

    absolute_values = np.maximum(np.abs(adjusted_profile), 1.0e-12)
    absolute_lags = np.abs(selected_lags).astype(float)
    sign_mismatch_fraction = float(np.mean(adjusted_profile <= 0.0))

    if absolute_values.size < 3:
        return TailFitResult(
            model_name=model_name,
            error=1.0e6,
            fitted_alpha=0.0,
            fitted_amplitude=0.0,
            oscillatory=oscillatory,
            sign_mismatch_fraction=sign_mismatch_fraction,
        )

    design_matrix = np.vstack([np.ones_like(absolute_lags), -absolute_lags]).T
    target_vector = np.log(absolute_values)
    fit_parameters, _, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)

    log_amplitude = float(fit_parameters[0])
    fitted_alpha = float(max(fit_parameters[1], 0.0))
    fitted_amplitude = float(np.exp(log_amplitude))
    fitted_values = fitted_amplitude * np.exp(-fitted_alpha * absolute_lags)

    root_mean_square_error = float(np.sqrt(np.mean((fitted_values - adjusted_profile) ** 2)))
    mean_absolute_level = float(np.mean(absolute_values) + 1.0e-14)
    relative_error = root_mean_square_error / mean_absolute_level + sign_mismatch_fraction

    return TailFitResult(
        model_name=model_name,
        error=relative_error,
        fitted_alpha=fitted_alpha,
        fitted_amplitude=fitted_amplitude,
        oscillatory=oscillatory,
        sign_mismatch_fraction=sign_mismatch_fraction,
    )


def fit_tail_models(lags: np.ndarray, profile: np.ndarray, min_tail_fit_lag: int) -> tuple[TailFitResult, TailFitResult, TailFitResult]:
    """Считает обе модели хвоста и выбирает лучшую."""
    monotonic_fit = fit_single_tail_model(
        lags=lags,
        profile=profile,
        min_tail_fit_lag=min_tail_fit_lag,
        oscillatory=False,
    )
    oscillatory_fit = fit_single_tail_model(
        lags=lags,
        profile=profile,
        min_tail_fit_lag=min_tail_fit_lag,
        oscillatory=True,
    )

    if oscillatory_fit.error <= monotonic_fit.error:
        best_fit = oscillatory_fit
    else:
        best_fit = monotonic_fit

    return monotonic_fit, oscillatory_fit, best_fit


def compute_recurrence_metric(
    time_grid: np.ndarray,
    profile_history: np.ndarray,
    late_indices: np.ndarray,
    dominant_period: float,
) -> tuple[int, float, float, float, np.ndarray, np.ndarray]:
    """Считает меру возврата профиля D(t) = ||x(t+T)-x(t)|| / ||x(t)||."""
    if not np.isfinite(dominant_period):
        return 0, float("nan"), float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    if time_grid.size < 2:
        return 0, float("nan"), float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    average_time_step = float(np.mean(np.diff(time_grid)))
    if average_time_step <= 0.0:
        return 0, float("nan"), float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    shift_steps = int(round(dominant_period / average_time_step))
    if shift_steps <= 0:
        return 0, float("nan"), float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    recurrence_times: list[float] = []
    recurrence_distances: list[float] = []

    for late_index in late_indices:
        shifted_index = int(late_index + shift_steps)
        if shifted_index >= profile_history.shape[0]:
            break

        reference_profile = profile_history[late_index]
        shifted_profile = profile_history[shifted_index]
        denominator = float(np.linalg.norm(reference_profile) + 1.0e-14)
        relative_distance = float(np.linalg.norm(shifted_profile - reference_profile) / denominator)

        recurrence_times.append(float(time_grid[late_index]))
        recurrence_distances.append(relative_distance)

    if not recurrence_distances:
        return shift_steps, shift_steps * average_time_step, float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    recurrence_times_array = np.asarray(recurrence_times, dtype=float)
    recurrence_distances_array = np.asarray(recurrence_distances, dtype=float)

    return (
        shift_steps,
        shift_steps * average_time_step,
        float(np.mean(recurrence_distances_array)),
        float(np.max(recurrence_distances_array)),
        recurrence_times_array,
        recurrence_distances_array,
    )


def compute_profile_diagnostics(
    profile_name: str,
    time_grid: np.ndarray,
    lags: np.ndarray,
    profile_history: np.ndarray,
    width_history: np.ndarray,
    central_history: np.ndarray,
    localization_history: np.ndarray,
    late_indices: np.ndarray,
    min_tail_fit_lag: int,
) -> tuple[ProfileDiagnostics, list[SpectrumMetric], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Считает полный набор численных диагностик для одного профиля."""
    zero_lag_index = get_lag_index(lags, 0)
    zero_lag_signal = profile_history[:, zero_lag_index]

    dominant_frequency, dominant_period, dominant_peak_ratio, recurrence_frequency_grid, recurrence_amplitude_spectrum = analyze_periodicity(
        time_grid=time_grid,
        signal=zero_lag_signal,
        late_indices=late_indices,
    )

    late_width = width_history[late_indices]
    late_central = central_history[late_indices]
    late_localization = localization_history[late_indices]
    late_mean_profile = np.mean(profile_history[late_indices], axis=0)

    minimum_toeplitz_value, passes_psd_check = check_profile_toeplitz_psd(lags, late_mean_profile)
    monotonic_fit, oscillatory_fit, best_fit = fit_tail_models(lags, late_mean_profile, min_tail_fit_lag)

    initial_central_amplitude = float(max(abs(central_history[0]), 1.0e-14))
    minimum_abs_late_central_amplitude = float(np.min(np.abs(late_central)))
    central_ratio = minimum_abs_late_central_amplitude / initial_central_amplitude

    recurrence_shift_steps, recurrence_time_shift, recurrence_mean_distance, recurrence_max_distance, recurrence_times, recurrence_distances = compute_recurrence_metric(
        time_grid=time_grid,
        profile_history=profile_history,
        late_indices=late_indices,
        dominant_period=dominant_period,
    )

    diagnostics = ProfileDiagnostics(
        profile_name=profile_name,
        saved_steps=int(time_grid.size),
        total_time=float(time_grid[-1]),
        lag_min=int(np.min(lags)),
        lag_max=int(np.max(lags)),
        initial_width=float(width_history[0]),
        final_width=float(width_history[-1]),
        min_late_width=float(np.min(late_width)),
        mean_late_width=float(np.mean(late_width)),
        max_late_width=float(np.max(late_width)),
        initial_central_amplitude=float(central_history[0]),
        final_central_amplitude=float(central_history[-1]),
        min_abs_late_central_amplitude=minimum_abs_late_central_amplitude,
        late_to_initial_central_amplitude_ratio=float(central_ratio),
        min_late_localization_fraction=float(np.min(late_localization)),
        mean_late_localization_fraction=float(np.mean(late_localization)),
        max_late_localization_fraction=float(np.max(late_localization)),
        dominant_frequency_lag_0=dominant_frequency,
        dominant_period_lag_0=dominant_period,
        dominant_peak_ratio_lag_0=dominant_peak_ratio,
        minimum_toeplitz_eigenvalue_on_late_mean_profile=minimum_toeplitz_value,
        passes_psd_check=passes_psd_check,
        monotonic_tail_alpha=monotonic_fit.fitted_alpha,
        monotonic_tail_error=monotonic_fit.error,
        oscillatory_tail_alpha=oscillatory_fit.fitted_alpha,
        oscillatory_tail_error=oscillatory_fit.error,
        oscillatory_tail_sign_mismatch_fraction=oscillatory_fit.sign_mismatch_fraction,
        best_tail_model=best_fit.model_name,
        best_tail_alpha=best_fit.fitted_alpha,
        best_tail_error=best_fit.error,
        recurrence_shift_steps=recurrence_shift_steps,
        recurrence_time_shift=recurrence_time_shift,
        recurrence_mean_relative_distance=recurrence_mean_distance,
        recurrence_max_relative_distance=recurrence_max_distance,
    )

    spectral_metrics: list[SpectrumMetric] = []
    spectral_payload = {
        "recurrence_frequency_grid": recurrence_frequency_grid,
        "recurrence_amplitude_spectrum": recurrence_amplitude_spectrum,
        "late_mean_profile": late_mean_profile,
        "recurrence_times": recurrence_times,
        "recurrence_distances": recurrence_distances,
    }
    return diagnostics, spectral_metrics, late_mean_profile, late_indices, spectral_payload


def compute_selected_spectrum_metrics(
    profile_name: str,
    time_grid: np.ndarray,
    lags: np.ndarray,
    profile_history: np.ndarray,
    late_indices: np.ndarray,
    selected_lags: list[int],
) -> tuple[list[SpectrumMetric], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Считает спектральные метрики для нескольких лагов."""
    spectral_metrics: list[SpectrumMetric] = []
    spectrum_data_by_lag: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for lag_value in selected_lags:
        lag_index = get_lag_index(lags, lag_value)
        signal = profile_history[:, lag_index]
        dominant_frequency, dominant_period, dominant_peak_ratio, frequency_grid, amplitude_spectrum = analyze_periodicity(
            time_grid=time_grid,
            signal=signal,
            late_indices=late_indices,
        )
        spectral_metrics.append(
            SpectrumMetric(
                profile_name=profile_name,
                lag=int(lag_value),
                dominant_frequency=dominant_frequency,
                dominant_period=dominant_period,
                dominant_peak_ratio=dominant_peak_ratio,
            )
        )
        spectrum_data_by_lag[int(lag_value)] = (frequency_grid, amplitude_spectrum)

    return spectral_metrics, spectrum_data_by_lag


def choose_tail_time_indices(late_indices: np.ndarray, tail_profile_count: int) -> np.ndarray:
    """Выбирает несколько характерных поздних моментов времени для хвостовых графиков."""
    if late_indices.size == 0:
        return np.asarray([], dtype=int)

    requested_count = int(max(tail_profile_count, 1))
    if late_indices.size <= requested_count:
        return late_indices.astype(int)

    floating_positions = np.linspace(0, late_indices.size - 1, requested_count)
    integer_positions = np.rint(floating_positions).astype(int)
    chosen_indices = late_indices[integer_positions]
    return np.unique(chosen_indices.astype(int))


def save_heatmap(
    output_path: Path,
    time_grid: np.ndarray,
    lags: np.ndarray,
    profile_history: np.ndarray,
    title: str,
    value_quantile: float,
) -> None:
    """Сохраняет heatmap эволюции профиля."""
    absolute_clip = float(np.quantile(np.abs(profile_history), value_quantile))
    if absolute_clip <= 0.0:
        absolute_clip = 1.0e-12

    figure, axes = plt.subplots(figsize=(9, 5))
    image = axes.imshow(
        profile_history.T,
        aspect="auto",
        origin="lower",
        extent=[float(time_grid[0]), float(time_grid[-1]), float(lags[0]), float(lags[-1])],
        vmin=-absolute_clip,
        vmax=absolute_clip,
    )
    axes.set_xlabel("t")
    axes.set_ylabel("лаг k")
    axes.set_title(title)
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_selected_spectra_plot(
    output_path: Path,
    spectrum_data_by_lag: dict[int, tuple[np.ndarray, np.ndarray]],
    title: str,
) -> None:
    """Сохраняет спектры для выбранных лагов."""
    figure, axes = plt.subplots(figsize=(9, 5))

    for lag_value in sorted(spectrum_data_by_lag):
        frequency_grid, amplitude_spectrum = spectrum_data_by_lag[lag_value]
        if frequency_grid.size <= 1:
            continue
        axes.plot(frequency_grid[1:], amplitude_spectrum[1:], label=f"k={lag_value}")

    axes.set_xlabel("частота")
    axes.set_ylabel("амплитуда спектра")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_tail_semilogy_plot(
    output_path: Path,
    time_grid: np.ndarray,
    lags: np.ndarray,
    profile_history: np.ndarray,
    late_indices: np.ndarray,
    late_mean_profile: np.ndarray,
    min_tail_fit_lag: int,
    tail_profile_count: int,
    title: str,
) -> None:
    """Сохраняет полулогарифмический график хвостов |profile_k|."""
    selected_time_indices = choose_tail_time_indices(late_indices, tail_profile_count)
    nonnegative_mask = lags >= 0
    absolute_lags = np.abs(lags[nonnegative_mask]).astype(float)

    figure, axes = plt.subplots(figsize=(9, 5))
    for profile_index in selected_time_indices:
        profile = np.abs(profile_history[profile_index][nonnegative_mask])
        axes.semilogy(absolute_lags, np.maximum(profile, 1.0e-12), label=f"t={time_grid[profile_index]:.3f}")

    mean_profile_absolute = np.abs(late_mean_profile[nonnegative_mask])
    axes.semilogy(
        absolute_lags,
        np.maximum(mean_profile_absolute, 1.0e-12),
        linewidth=2.0,
        label="late mean",
    )

    monotonic_fit = fit_single_tail_model(lags, late_mean_profile, min_tail_fit_lag, oscillatory=False)
    fit_mask = absolute_lags >= float(min_tail_fit_lag)
    fitted_tail = monotonic_fit.fitted_amplitude * np.exp(-monotonic_fit.fitted_alpha * absolute_lags[fit_mask])
    axes.semilogy(
        absolute_lags[fit_mask],
        np.maximum(fitted_tail, 1.0e-12),
        linestyle="--",
        linewidth=2.0,
        label=f"fit alpha={monotonic_fit.fitted_alpha:.4f}",
    )

    axes.set_xlabel("|k|")
    axes.set_ylabel("|profile_k|")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_recurrence_plot(
    output_path: Path,
    recurrence_times: np.ndarray,
    recurrence_distances: np.ndarray,
    title: str,
) -> None:
    """Сохраняет график меры возврата D(t)."""
    figure, axes = plt.subplots(figsize=(9, 4.5))
    if recurrence_times.size > 0:
        axes.plot(recurrence_times, recurrence_distances)
    axes.set_xlabel("t")
    axes.set_ylabel("D(t)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_profile_metrics_csv(output_path: Path, diagnostics_rows: list[ProfileDiagnostics]) -> None:
    """Записывает широкую таблицу метрик по профилям xi и kappa."""
    field_names = list(ProfileDiagnostics.__annotations__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for diagnostics_row in diagnostics_rows:
            writer.writerow(asdict(diagnostics_row))


def write_spectrum_metrics_csv(output_path: Path, spectral_rows: list[SpectrumMetric]) -> None:
    """Записывает спектральные метрики по выбранным лагам."""
    field_names = list(SpectrumMetric.__annotations__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for spectral_row in spectral_rows:
            writer.writerow(asdict(spectral_row))


def write_metrics_json(
    output_path: Path,
    diagnostics_rows: list[ProfileDiagnostics],
    spectral_rows: list[SpectrumMetric],
    config: DiagnosticsConfig,
) -> None:
    """Сохраняет ту же диагностику в json."""
    payload = {
        "npz_path": str(config.npz_path),
        "late_time_fraction": float(config.late_time_fraction),
        "min_tail_fit_lag": int(config.min_tail_fit_lag),
        "spectrum_lags": list(config.spectrum_lags or []),
        "profiles": [asdict(diagnostics_row) for diagnostics_row in diagnostics_rows],
        "spectra": [asdict(spectral_row) for spectral_row in spectral_rows],
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def format_float(value: float) -> str:
    """Форматирует число для печати табличной сводки."""
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def print_profile_summary_table(diagnostics_rows: list[ProfileDiagnostics]) -> None:
    """Печатает короткую сводную таблицу в консоль."""
    headers = [
        "profile",
        "final_width",
        "max_late_width",
        "central_ratio",
        "freq_k0",
        "peak_ratio_k0",
        "tail_model",
        "tail_alpha",
        "tail_error",
        "recurrence_mean_D",
    ]
    row_format = "{:<8} {:>12} {:>14} {:>14} {:>12} {:>15} {:>12} {:>12} {:>12} {:>18}"

    print(row_format.format(*headers))
    for diagnostics_row in diagnostics_rows:
        print(
            row_format.format(
                diagnostics_row.profile_name,
                format_float(diagnostics_row.final_width),
                format_float(diagnostics_row.max_late_width),
                format_float(diagnostics_row.late_to_initial_central_amplitude_ratio),
                format_float(diagnostics_row.dominant_frequency_lag_0),
                format_float(diagnostics_row.dominant_peak_ratio_lag_0),
                diagnostics_row.best_tail_model,
                format_float(diagnostics_row.best_tail_alpha),
                format_float(diagnostics_row.best_tail_error),
                format_float(diagnostics_row.recurrence_mean_relative_distance),
            )
        )


def run_diagnostics(config: DiagnosticsConfig) -> None:
    """Запускает полный постпроцессинг по сохранённому time_series.npz."""
    ensure_output_directory(config.output_directory)
    arrays = load_time_series(config.npz_path)

    time_grid = arrays["time_grid"]
    lags = arrays["lags"]
    xi_history = arrays["xi_history"]
    kappa_history = arrays["kappa_history"]
    xi_width_history = arrays["xi_width_history"]
    kappa_width_history = arrays["kappa_width_history"]
    xi_central_history = arrays["xi_central_amplitude_history"]
    kappa_central_history = arrays["kappa_central_amplitude_history"]
    xi_localization_history = arrays["xi_localization_history"]
    kappa_localization_history = arrays["kappa_localization_history"]

    late_indices = get_late_time_indices(time_grid, config.late_time_fraction)
    selected_lags = get_available_spectrum_lags(lags, config.spectrum_lags or [0, 1, 2, 3])

    profile_diagnostics_rows: list[ProfileDiagnostics] = []
    spectrum_metric_rows: list[SpectrumMetric] = []

    xi_diagnostics, _, xi_late_mean_profile, _, xi_payload = compute_profile_diagnostics(
        profile_name="xi",
        time_grid=time_grid,
        lags=lags,
        profile_history=xi_history,
        width_history=xi_width_history,
        central_history=xi_central_history,
        localization_history=xi_localization_history,
        late_indices=late_indices,
        min_tail_fit_lag=config.min_tail_fit_lag,
    )
    xi_spectra, xi_spectrum_data = compute_selected_spectrum_metrics(
        profile_name="xi",
        time_grid=time_grid,
        lags=lags,
        profile_history=xi_history,
        late_indices=late_indices,
        selected_lags=selected_lags,
    )
    profile_diagnostics_rows.append(xi_diagnostics)
    spectrum_metric_rows.extend(xi_spectra)

    kappa_diagnostics, _, kappa_late_mean_profile, _, kappa_payload = compute_profile_diagnostics(
        profile_name="kappa",
        time_grid=time_grid,
        lags=lags,
        profile_history=kappa_history,
        width_history=kappa_width_history,
        central_history=kappa_central_history,
        localization_history=kappa_localization_history,
        late_indices=late_indices,
        min_tail_fit_lag=config.min_tail_fit_lag,
    )
    kappa_spectra, kappa_spectrum_data = compute_selected_spectrum_metrics(
        profile_name="kappa",
        time_grid=time_grid,
        lags=lags,
        profile_history=kappa_history,
        late_indices=late_indices,
        selected_lags=selected_lags,
    )
    profile_diagnostics_rows.append(kappa_diagnostics)
    spectrum_metric_rows.extend(kappa_spectra)

    write_profile_metrics_csv(config.output_directory / "profile_metrics.csv", profile_diagnostics_rows)
    write_spectrum_metrics_csv(config.output_directory / "spectrum_metrics.csv", spectrum_metric_rows)
    write_metrics_json(config.output_directory / "diagnostics.json", profile_diagnostics_rows, spectrum_metric_rows, config)

    save_heatmap(
        output_path=config.output_directory / "xi_heatmap.png",
        time_grid=time_grid,
        lags=lags,
        profile_history=xi_history,
        title="Эволюция профиля xi_k(t)",
        value_quantile=config.heatmap_value_quantile,
    )
    save_heatmap(
        output_path=config.output_directory / "kappa_heatmap.png",
        time_grid=time_grid,
        lags=lags,
        profile_history=kappa_history,
        title="Эволюция профиля kappa_k(t)",
        value_quantile=config.heatmap_value_quantile,
    )

    save_selected_spectra_plot(
        output_path=config.output_directory / "xi_selected_lag_spectra.png",
        spectrum_data_by_lag=xi_spectrum_data,
        title="Поздние спектры xi_k(t) для нескольких лагов",
    )
    save_selected_spectra_plot(
        output_path=config.output_directory / "kappa_selected_lag_spectra.png",
        spectrum_data_by_lag=kappa_spectrum_data,
        title="Поздние спектры kappa_k(t) для нескольких лагов",
    )

    save_tail_semilogy_plot(
        output_path=config.output_directory / "xi_tail_semilogy.png",
        time_grid=time_grid,
        lags=lags,
        profile_history=xi_history,
        late_indices=late_indices,
        late_mean_profile=xi_late_mean_profile,
        min_tail_fit_lag=config.min_tail_fit_lag,
        tail_profile_count=config.tail_profile_count,
        title="Полулогарифмический хвост xi_k",
    )
    save_tail_semilogy_plot(
        output_path=config.output_directory / "kappa_tail_semilogy.png",
        time_grid=time_grid,
        lags=lags,
        profile_history=kappa_history,
        late_indices=late_indices,
        late_mean_profile=kappa_late_mean_profile,
        min_tail_fit_lag=config.min_tail_fit_lag,
        tail_profile_count=config.tail_profile_count,
        title="Полулогарифмический хвост kappa_k",
    )

    save_recurrence_plot(
        output_path=config.output_directory / "xi_recurrence.png",
        recurrence_times=xi_payload["recurrence_times"],
        recurrence_distances=xi_payload["recurrence_distances"],
        title="Мера возврата профиля xi_k",
    )
    save_recurrence_plot(
        output_path=config.output_directory / "kappa_recurrence.png",
        recurrence_times=kappa_payload["recurrence_times"],
        recurrence_distances=kappa_payload["recurrence_distances"],
        title="Мера возврата профиля kappa_k",
    )

    print_profile_summary_table(profile_diagnostics_rows)
    print(f"Диагностика сохранена в {config.output_directory}")


def main(
    npz_path: str | Path = "results/time_series.npz",
    output_directory: str | Path | None = None,
    late_time_fraction: float = 0.5,
    min_tail_fit_lag: int = 4,
    spectrum_lags: list[int] | None = None,
    heatmap_value_quantile: float = 0.995,
    tail_profile_count: int = 3,
) -> None:
    """Точка входа для автономного запуска диагностики."""
    config = build_default_config(
        npz_path=npz_path,
        output_directory=output_directory,
        late_time_fraction=late_time_fraction,
        min_tail_fit_lag=min_tail_fit_lag,
        spectrum_lags=spectrum_lags,
        heatmap_value_quantile=heatmap_value_quantile,
        tail_profile_count=tail_profile_count,
    )
    run_diagnostics(config)


if __name__ == "__main__":
    main()
