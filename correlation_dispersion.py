from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from correlations import compute_correlation_profiles
from integrator import velocity_verlet_step
from io_utils import read_config


@dataclass(slots=True)
class DispersionRunConfig:
    config_path: Path
    output_directory: Path
    amplitudes: list[float]
    mode_indices: list[int]
    total_steps: int
    save_every_steps: int
    time_step: float
    chain_length: int
    mass: float
    stiffness: float
    beta: float
    max_lag: int
    ensemble_size: int
    random_seed: int
    zero_displacements: bool
    spatial_averaging: bool
    background_value: float
    covariance_psd_tolerance: float
    covariance_negative_warning_tolerance: float


@dataclass(slots=True)
class ModeMeasurement:
    amplitude: float
    mode_index: int
    wave_number: float
    measured_frequency: float
    dominant_peak_ratio: float
    early_frequency: float
    early_peak_ratio: float
    projection_norm: float
    minimum_spectrum_value: float


class CorrelationModeExperiment:
    def __init__(self, config: DispersionRunConfig) -> None:
        self.config = config
        self.lags = np.arange(-config.max_lag, config.max_lag + 1, dtype=int)
        self.lag_float = self.lags.astype(float)
        self.full_lags = np.arange(config.chain_length, dtype=int)
        self.full_lag_float = self.full_lags.astype(float)

    def run(self) -> None:
        self.config.output_directory.mkdir(parents=True, exist_ok=True)

        all_measurements: list[ModeMeasurement] = []
        series_payload: dict[str, dict[str, list[float]]] = {}

        for amplitude in self.config.amplitudes:
            amplitude_key = self._format_amplitude_key(amplitude)
            series_payload[amplitude_key] = {}

            for mode_index in self.config.mode_indices:
                measurement, time_grid, projection_series = self._run_single_mode(amplitude, mode_index)
                all_measurements.append(measurement)
                series_payload[amplitude_key][str(mode_index)] = projection_series.tolist()
                self._save_single_series_plot(amplitude, mode_index, time_grid, projection_series)

        self._save_measurements_csv(all_measurements)
        self._save_measurements_json(all_measurements, series_payload)
        self._save_dispersion_plot(all_measurements)

    def _run_single_mode(
        self,
        amplitude: float,
        mode_index: int,
    ) -> tuple[ModeMeasurement, np.ndarray, np.ndarray]:
        displacement_ensemble, velocity_ensemble, minimum_spectrum_value = self._create_mode_ensemble(
            amplitude,
            mode_index,
        )

        saved_times: list[float] = []
        saved_projections: list[float] = []

        wave_number = 2.0 * np.pi * mode_index / self.config.chain_length
        cosine_profile = np.cos(wave_number * self.lag_float)
        denominator = float(np.sum(cosine_profile ** 2))
        if denominator <= 1.0e-14:
            raise ValueError(f"Не удалось построить моду для mode_index={mode_index}")

        for step_index in range(self.config.total_steps + 1):
            if step_index % self.config.save_every_steps == 0:
                _, kappa_profile = compute_correlation_profiles(
                    displacement_ensemble=displacement_ensemble,
                    velocity_ensemble=velocity_ensemble,
                    lags=self.lags,
                    spatial_averaging=self.config.spatial_averaging,
                )
                kappa_profile_without_background = kappa_profile - float(np.mean(kappa_profile))
                projection_value = float(
                    np.dot(kappa_profile_without_background, cosine_profile) / denominator
                )
                saved_times.append(step_index * self.config.time_step)
                saved_projections.append(projection_value)

            if step_index < self.config.total_steps:
                displacement_ensemble, velocity_ensemble = velocity_verlet_step(
                    displacement_ensemble=displacement_ensemble,
                    velocity_ensemble=velocity_ensemble,
                    time_step=self.config.time_step,
                    stiffness=self.config.stiffness,
                    beta=self.config.beta,
                    mass=self.config.mass,
                )

        time_grid = np.asarray(saved_times, dtype=float)
        projection_series = np.asarray(saved_projections, dtype=float)
        measured_frequency, dominant_peak_ratio = self._measure_frequency(time_grid, projection_series)

        measurement = ModeMeasurement(
            amplitude=float(amplitude),
            mode_index=int(mode_index),
            wave_number=float(wave_number),
            measured_frequency=measured_frequency,
            dominant_peak_ratio=dominant_peak_ratio,
            projection_norm=float(np.linalg.norm(projection_series)),
            minimum_spectrum_value=float(minimum_spectrum_value),
        )
        return measurement, time_grid, projection_series

    def _create_mode_ensemble(
        self,
        amplitude: float,
        mode_index: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ensemble_shape = (self.config.ensemble_size, self.config.chain_length)
        displacement_ensemble = np.zeros(ensemble_shape, dtype=float)
        if not self.config.zero_displacements:
            displacement_ensemble = np.random.normal(0.0, 1.0, size=ensemble_shape)

        wave_number = 2.0 * np.pi * mode_index / self.config.chain_length
        covariance_row = self._build_covariance_row(amplitude, wave_number)
        spectrum = np.real(np.fft.fft(covariance_row))
        minimum_spectrum_value = float(np.min(spectrum))
        clipped_spectrum = self._validate_and_prepare_spectrum(spectrum)
        filter_amplitudes = np.sqrt(np.maximum(clipped_spectrum, 0.0))

        white_noise = np.random.normal(0.0, 1.0, size=ensemble_shape)
        white_noise_spectrum = np.fft.fft(white_noise, axis=1)
        shaped_spectrum = white_noise_spectrum * filter_amplitudes[None, :]
        velocity_ensemble = np.real(np.fft.ifft(shaped_spectrum, axis=1))

        return displacement_ensemble, velocity_ensemble, minimum_spectrum_value

    def _build_covariance_row(self, amplitude: float, wave_number: float) -> np.ndarray:
        covariance_row = np.full(self.config.chain_length, self.config.background_value, dtype=float)
        covariance_row += amplitude * np.cos(wave_number * self.full_lag_float)
        return covariance_row

    def _validate_and_prepare_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        minimum_spectrum_value = float(np.min(spectrum))

        if minimum_spectrum_value >= -self.config.covariance_psd_tolerance:
            return np.clip(spectrum, a_min=0.0, a_max=None)

        if minimum_spectrum_value >= -self.config.covariance_negative_warning_tolerance:
            return np.clip(spectrum, a_min=0.0, a_max=None)

        raise ValueError(
            "Спектральная плотность заданной корреляционной моды отрицательна: "
            f"minimum={minimum_spectrum_value:.3e}"
        )

    def _measure_frequency(self, time_grid: np.ndarray, series: np.ndarray) -> tuple[float, float]:
        if time_grid.size < 4:
            return 0.0, 0.0

        detrended_series = series - float(np.mean(series))
        average_time_step = float(np.mean(np.diff(time_grid)))
        if average_time_step <= 0.0:
            return 0.0, 0.0

        frequency_grid = np.fft.rfftfreq(detrended_series.size, d=average_time_step)
        amplitude_spectrum = np.abs(np.fft.rfft(detrended_series))
        if amplitude_spectrum.size <= 1:
            return 0.0, 0.0

        positive_frequencies = frequency_grid[1:]
        positive_amplitudes = amplitude_spectrum[1:]

        peak_index = int(np.argmax(positive_amplitudes))
        dominant_amplitude = float(positive_amplitudes[peak_index])
        background = float(np.median(positive_amplitudes) + 1.0e-14)
        dominant_peak_ratio = dominant_amplitude / background

        # Параболическая интерполяция пика
        if 0 < peak_index < positive_amplitudes.size - 1:
            alpha_left = float(positive_amplitudes[peak_index - 1])
            alpha_center = float(positive_amplitudes[peak_index])
            alpha_right = float(positive_amplitudes[peak_index + 1])
            denominator_interp = alpha_left - 2.0 * alpha_center + alpha_right
            if abs(denominator_interp) > 1.0e-14:
                delta = 0.5 * (alpha_left - alpha_right) / denominator_interp
                freq_step = float(positive_frequencies[1] - positive_frequencies[0])
                dominant_frequency = float(positive_frequencies[peak_index]) + delta * freq_step
            else:
                dominant_frequency = float(positive_frequencies[peak_index])
        else:
            dominant_frequency = float(positive_frequencies[peak_index])

        return dominant_frequency, dominant_peak_ratio

    def _save_single_series_plot(
        self,
        amplitude: float,
        mode_index: int,
        time_grid: np.ndarray,
        projection_series: np.ndarray,
    ) -> None:
        figure, axes = plt.subplots(figsize=(8, 4.5))
        axes.plot(time_grid, projection_series)
        axes.set_xlabel("t")
        axes.set_ylabel("A_q(t)")
        axes.set_title(
            f"Проекция корреляционной моды: amp={amplitude:.4g}, m={mode_index}"
        )
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        output_path = self.config.output_directory / (
            f"projection_amp_{self._format_amplitude_key(amplitude)}_mode_{mode_index}.png"
        )
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    def _save_dispersion_plot(self, measurements: list[ModeMeasurement]) -> None:
        figure, axes = plt.subplots(figsize=(8, 5))

        amplitude_values = sorted({measurement.amplitude for measurement in measurements})
        for amplitude in amplitude_values:
            selected = [m for m in measurements if m.amplitude == amplitude]
            selected.sort(key=lambda item: item.wave_number)
            wave_numbers = [item.wave_number for item in selected]
            frequencies = [item.measured_frequency for item in selected]
            axes.plot(wave_numbers, frequencies, marker="o", label=f"amp={amplitude:.4g}")

        axes.set_xlabel("q")
        axes.set_ylabel("Omega(q)")
        axes.set_title("Численная дисперсионная кривая в пространстве ковариаций")
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()
        figure.savefig(self.config.output_directory / "dispersion_curves.png", dpi=160)
        plt.close(figure)

    def _save_measurements_csv(self, measurements: list[ModeMeasurement]) -> None:
        output_path = self.config.output_directory / "dispersion_measurements.csv"
        with output_path.open("w", encoding="utf-8") as stream:
            stream.write(
                "amplitude,mode_index,wave_number,measured_frequency,dominant_peak_ratio,projection_norm,minimum_spectrum_value\n"
            )
            for measurement in measurements:
                stream.write(
                    f"{measurement.amplitude},"
                    f"{measurement.mode_index},"
                    f"{measurement.wave_number},"
                    f"{measurement.measured_frequency},"
                    f"{measurement.dominant_peak_ratio},"
                    f"{measurement.projection_norm},"
                    f"{measurement.minimum_spectrum_value}\n"
                )

    def _save_measurements_json(
        self,
        measurements: list[ModeMeasurement],
        series_payload: dict[str, dict[str, list[float]]],
    ) -> None:
        payload = {
            "config_path": str(self.config.config_path),
            "output_directory": str(self.config.output_directory),
            "background_value": self.config.background_value,
            "amplitudes": self.config.amplitudes,
            "mode_indices": self.config.mode_indices,
            "measurements": [
                {
                    "amplitude": measurement.amplitude,
                    "mode_index": measurement.mode_index,
                    "wave_number": measurement.wave_number,
                    "measured_frequency": measurement.measured_frequency,
                    "dominant_peak_ratio": measurement.dominant_peak_ratio,
                    "projection_norm": measurement.projection_norm,
                    "minimum_spectrum_value": measurement.minimum_spectrum_value,
                }
                for measurement in measurements
            ],
            "projection_series": series_payload,
        }
        output_path = self.config.output_directory / "dispersion_measurements.json"
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    def _format_amplitude_key(self, amplitude: float) -> str:
        return f"{amplitude:.6f}".replace(".", "p")


def load_dispersion_run_config(
    config_path: str | Path = "config.yaml",
    output_directory: str | Path = "dispersion_results",
    amplitudes: list[float] | None = None,
    mode_indices: list[int] | None = None,
    total_steps: int | None = None,
    save_every_steps: int | None = None,
    background_value: float = 1.0,
    max_lag: int | None = None,
    spatial_averaging: bool | None = None,
    random_seed: int | None = None,
) -> DispersionRunConfig:
    config_path_obj = Path(config_path)
    raw_config = read_config(config_path_obj)

    experiment_settings = raw_config["experiment"]
    chain_settings = raw_config["chain"]
    correlation_settings = raw_config["correlations"]
    initial_settings = raw_config["initial_conditions"]

    if amplitudes is None:
        amplitudes = [0.02, 0.05, 0.1, 0.2]
    if mode_indices is None:
        mode_indices = list(range(1, 17))

    effective_total_steps = int(
        experiment_settings["total_steps"] if total_steps is None else total_steps
    )
    effective_save_every_steps = int(
        experiment_settings["save_every_steps"] if save_every_steps is None else save_every_steps
    )
    effective_max_lag = int(
        correlation_settings["max_lag"] if max_lag is None else max_lag
    )
    effective_spatial_averaging = bool(
        initial_settings.get("spatial_averaging", True)
        if spatial_averaging is None
        else spatial_averaging
    )
    effective_random_seed = int(
        experiment_settings["random_seed"] if random_seed is None else random_seed
    )

    return DispersionRunConfig(
        config_path=config_path_obj,
        output_directory=Path(output_directory),
        amplitudes=list(amplitudes),
        mode_indices=list(mode_indices),
        total_steps=effective_total_steps,
        save_every_steps=effective_save_every_steps,
        time_step=float(chain_settings["dt"]),
        chain_length=int(chain_settings["length"]),
        mass=float(chain_settings["mass"]),
        stiffness=float(chain_settings["stiffness"]),
        beta=float(chain_settings["beta"]),
        max_lag=effective_max_lag,
        ensemble_size=int(experiment_settings["ensemble_size"]),
        random_seed=effective_random_seed,
        zero_displacements=bool(initial_settings.get("zero_displacements", True)),
        spatial_averaging=effective_spatial_averaging,
        background_value=float(background_value),
        covariance_psd_tolerance=float(
            initial_settings.get("covariance_psd_tolerance", 1.0e-10)
        ),
        covariance_negative_warning_tolerance=float(
            initial_settings.get("covariance_negative_warning_tolerance", 1.0e-8)
        ),
    )


def main(
    config_path: str | Path = "config.yaml",
    output_directory: str | Path = "dispersion_results",
    amplitudes: list[float] | None = None,
    mode_indices: list[int] | None = None,
    total_steps: int | None = None,
    save_every_steps: int | None = None,
    background_value: float = 1.0,
    max_lag: int | None = None,
    spatial_averaging: bool | None = None,
    random_seed: int | None = 42,
) -> None:
    config = load_dispersion_run_config(
        config_path=config_path,
        output_directory=output_directory,
        amplitudes=amplitudes,
        mode_indices=mode_indices,
        total_steps=total_steps,
        save_every_steps=save_every_steps,
        background_value=background_value,
        max_lag=max_lag,
        spatial_averaging=spatial_averaging,
        random_seed=random_seed,
    )
    np.random.seed(config.random_seed)
    experiment = CorrelationModeExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
