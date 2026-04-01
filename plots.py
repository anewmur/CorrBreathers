from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from detector import get_late_time_indices


def save_diagnostic_plots(
    plots_enabled: bool,
    output_directory: Path,
    time_grid: np.ndarray,
    lags: np.ndarray,
    xi_history: np.ndarray,
    kappa_history: np.ndarray,
    xi_width_history: np.ndarray,
    kappa_width_history: np.ndarray,
    xi_central_amplitude_history: np.ndarray,
    kappa_central_amplitude_history: np.ndarray,
    detector_settings: dict,
    detector_result: dict,
) -> None:
    """Сохраняет набор диагностических графиков."""
    if not plots_enabled:
        return

    save_profile_plot(
        output_directory / "xi_profile_initial.png",
        lags,
        xi_history[0],
        "Начальный профиль xi",
        "xi_k",
    )
    save_profile_plot(
        output_directory / "xi_profile_final.png",
        lags,
        xi_history[-1],
        "Конечный профиль xi",
        "xi_k",
    )
    save_time_series_plot(
        output_directory / "xi0_time_series.png",
        time_grid,
        xi_central_amplitude_history,
        "xi_0(t)",
        "xi_0",
    )
    save_late_spectrum_plot(
        output_directory / "xi0_late_spectrum.png",
        time_grid,
        xi_central_amplitude_history,
        detector_settings,
        detector_result["xi_metrics"],
        "Late-window spectrum: xi_0(t)",
    )
    save_time_series_plot(
        output_directory / "xi_width_time_series.png",
        time_grid,
        xi_width_history,
        "Ширина профиля xi",
        "ширина xi",
    )

    save_profile_plot(
        output_directory / "kappa_profile_initial.png",
        lags,
        kappa_history[0],
        "Начальный профиль kappa",
        "kappa_k",
    )
    save_profile_plot(
        output_directory / "kappa_profile_final.png",
        lags,
        kappa_history[-1],
        "Конечный профиль kappa",
        "kappa_k",
    )
    save_time_series_plot(
        output_directory / "kappa0_time_series.png",
        time_grid,
        kappa_central_amplitude_history,
        "kappa_0(t)",
        "kappa_0",
    )
    save_late_spectrum_plot(
        output_directory / "kappa0_late_spectrum.png",
        time_grid,
        kappa_central_amplitude_history,
        detector_settings,
        detector_result["kappa_metrics"],
        "Late-window spectrum: kappa_0(t)",
    )
    save_time_series_plot(
        output_directory / "kappa_width_time_series.png",
        time_grid,
        kappa_width_history,
        "Ширина профиля kappa",
        "ширина kappa",
    )


def save_profile_plot(
    output_path: Path,
    lags: np.ndarray,
    profile_values: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """Сохраняет график профиля по лагу."""
    plt.figure(figsize=(7, 4))
    plt.plot(lags, profile_values, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("лаг k")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_time_series_plot(
    output_path: Path,
    time_grid: np.ndarray,
    series_values: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    """Сохраняет график временного ряда."""
    plt.figure(figsize=(7, 4))
    plt.plot(time_grid, series_values, linewidth=1.5)
    plt.xlabel("время")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_late_window_spectrum(
    time_grid: np.ndarray,
    signal_values: np.ndarray,
    detector_settings: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит спектр только по late-window, согласованный с детектором."""
    if time_grid.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    late_indices = get_late_time_indices(time_grid, detector_settings)
    late_signal = signal_values[late_indices]
    detrended_signal = late_signal - np.mean(late_signal)

    if detrended_signal.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    late_time_grid = time_grid[late_indices]
    time_step = float(np.mean(np.diff(late_time_grid)))
    frequency_grid = np.fft.rfftfreq(detrended_signal.size, d=time_step)
    amplitude_spectrum = np.abs(np.fft.rfft(detrended_signal))
    return frequency_grid, amplitude_spectrum


def get_metrics_dominant_frequency(metrics_block: dict) -> float:
    """Возвращает частоту из блока метрик при допустимых вариантах имени ключа."""
    if "dominant_frequency" in metrics_block:
        return float(metrics_block["dominant_frequency"])
    if "dominant_frequency_late" in metrics_block:
        return float(metrics_block["dominant_frequency_late"])
    return 0.0


def save_late_spectrum_plot(
    output_path: Path,
    time_grid: np.ndarray,
    signal_values: np.ndarray,
    detector_settings: dict,
    metrics_block: dict,
    title: str,
) -> None:
    """Сохраняет late-window spectrum и отмечает главный пик."""
    frequency_grid, amplitude_spectrum = compute_late_window_spectrum(
        time_grid,
        signal_values,
        detector_settings,
    )
    if frequency_grid.size == 0 or amplitude_spectrum.size == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(frequency_grid, amplitude_spectrum, linewidth=1.5)

    dominant_frequency = get_metrics_dominant_frequency(metrics_block)
    if dominant_frequency > 0.0:
        plt.axvline(dominant_frequency, color="red", linestyle="--", label="detector dominant frequency")
        plt.legend()

    plt.xlabel("частота")
    plt.ylabel("амплитуда")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()