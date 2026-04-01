from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_diagnostic_plots(
    plots_enabled: bool,
    output_directory: Path,
    time_grid: np.ndarray,
    lags: np.ndarray,
    kappa_history: np.ndarray,
    width_history: np.ndarray,
    central_amplitude_history: np.ndarray,
    detector_result: dict,
) -> None:
    """Сохраняет набор диагностических графиков."""
    if not plots_enabled:
        return

    save_profile_plot(
        output_directory / "kappa_profile_initial.png",
        lags,
        kappa_history[0],
        "Начальный профиль kappa",
    )
    save_profile_plot(
        output_directory / "kappa_profile_final.png",
        lags,
        kappa_history[-1],
        "Конечный профиль kappa",
    )
    save_time_series_plot(
        output_directory / "kappa0_time_series.png",
        time_grid,
        central_amplitude_history,
        "kappa_0(t)",
        "kappa_0",
    )
    save_spectrum_plot(
        output_directory / "kappa0_spectrum.png",
        time_grid,
        central_amplitude_history,
        detector_result,
    )
    save_time_series_plot(
        output_directory / "width_time_series.png",
        time_grid,
        width_history,
        "Ширина профиля kappa",
        "ширина",
    )


def save_profile_plot(
    output_path: Path,
    lags: np.ndarray,
    profile_values: np.ndarray,
    title: str,
) -> None:
    """Сохраняет график профиля по лагу."""
    plt.figure(figsize=(7, 4))
    plt.plot(lags, profile_values, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("лаг k")
    plt.ylabel("kappa_k")
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


def save_spectrum_plot(
    output_path: Path,
    time_grid: np.ndarray,
    kappa_zero_series: np.ndarray,
    detector_result: dict,
) -> None:
    """Сохраняет спектр kappa_0(t) и отмечает главный пик."""
    if time_grid.size < 4:
        return

    time_step = float(np.mean(np.diff(time_grid)))
    detrended_signal = kappa_zero_series - np.mean(kappa_zero_series)
    frequency_grid = np.fft.rfftfreq(detrended_signal.size, d=time_step)
    amplitude_spectrum = np.abs(np.fft.rfft(detrended_signal))

    plt.figure(figsize=(7, 4))
    plt.plot(frequency_grid, amplitude_spectrum, linewidth=1.5)
    dominant_frequency = float(detector_result.get("detected_frequency", 0.0))
    if dominant_frequency > 0.0:
        plt.axvline(dominant_frequency, color="red", linestyle="--", label="обнаруженный пик")
        plt.legend()
    plt.xlabel("частота")
    plt.ylabel("амплитуда")
    plt.title("Спектр kappa_0(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
