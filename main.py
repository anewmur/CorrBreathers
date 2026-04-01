from __future__ import annotations

from pathlib import Path

import numpy as np

from correlations import (
    compute_correlation_profiles,
    compute_correlation_width,
    compute_localization_fraction,
)
from detector import evaluate_breather_candidate
from init_conditions import create_initial_ensemble
from integrator import velocity_verlet_step
from io_utils import (
    ensure_output_directory,
    print_progress,
    read_config,
    save_breather_json,
    save_summary_text,
    save_time_series_npz,
)
from model import compute_total_energy_ensemble
from plots import save_diagnostic_plots


def parse_experiment_config(config_path: Path) -> tuple[dict, dict, Path]:
    """Читает конфиг и возвращает разделы для расчёта."""
    config = read_config(config_path)
    sections = {
        "experiment": config["experiment"],
        "chain": config["chain"],
        "initial_conditions": config["initial_conditions"],
        "correlations": config["correlations"],
        "detector": config["detector"],
        "plots": config["plots"],
        "numerics": config["numerics"],
    }
    output_directory = ensure_output_directory(config)
    return config, sections, output_directory


def prepare_simulation_parameters(sections: dict) -> dict:
    """Готовит численные параметры и массив лагов."""
    experiment_settings = sections["experiment"]
    chain_settings = sections["chain"]
    correlation_settings = sections["correlations"]

    max_lag = int(correlation_settings["max_lag"])
    parameters = {
        "random_seed": int(experiment_settings["random_seed"]),
        "ensemble_size": int(experiment_settings["ensemble_size"]),
        "total_steps": int(experiment_settings["total_steps"]),
        "save_every_steps": int(experiment_settings["save_every_steps"]),
        "chain_length": int(chain_settings["length"]),
        "mass": float(chain_settings["mass"]),
        "stiffness": float(chain_settings["stiffness"]),
        "beta": float(chain_settings["beta"]),
        "time_step": float(chain_settings["dt"]),
        "localization_radius": int(correlation_settings["localization_radius"]),
        "max_relative_energy_drift_warning": float(
            sections["numerics"]["max_relative_energy_drift_warning"]
        ),
        "lags": np.arange(-max_lag, max_lag + 1, dtype=int),
        "max_lag": max_lag,
    }
    np.random.seed(parameters["random_seed"])
    return parameters


def create_starting_ensemble(parameters: dict, initial_conditions_settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт начальный ансамбль смещений и скоростей."""
    return create_initial_ensemble(
        ensemble_size=parameters["ensemble_size"],
        chain_length=parameters["chain_length"],
        mode=str(initial_conditions_settings["mode"]),
        zero_displacements=bool(initial_conditions_settings["zero_displacements"]),
        random_thermal_std=float(initial_conditions_settings["random_thermal_std"]),
        amplitude=float(initial_conditions_settings["amplitude"]),
        alpha=float(initial_conditions_settings["alpha"]),
        covariance_psd_tolerance=float(initial_conditions_settings["covariance_psd_tolerance"]),
        covariance_negative_warning_tolerance=float(
            initial_conditions_settings["covariance_negative_warning_tolerance"]
        ),
    )


def compute_relative_energy_drift(energy_history: list[float], average_energy: float) -> float:
    """Вычисляет относительный дрейф энергии от первого сохранения."""
    if not energy_history:
        return 0.0
    initial_energy = max(abs(energy_history[0]), 1.0e-14)
    return float((average_energy - energy_history[0]) / initial_energy)


def collect_step_metrics(
    parameters: dict,
    step_index: int,
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    energy_history: list[float],
) -> dict:
    """Считает метрики на одном шаге сохранения."""
    current_time = step_index * parameters["time_step"]
    xi_profile, kappa_profile = compute_correlation_profiles(
        displacement_ensemble,
        velocity_ensemble,
        parameters["lags"],
    )
    average_energy = compute_total_energy_ensemble(
        displacement_ensemble,
        velocity_ensemble,
        parameters["mass"],
        parameters["stiffness"],
        parameters["beta"],
    )
    return {
        "current_time": current_time,
        "xi_profile": xi_profile,
        "kappa_profile": kappa_profile,
        "profile_width": compute_correlation_width(kappa_profile, parameters["lags"]),
        "central_amplitude": float(kappa_profile[parameters["max_lag"]]),
        "localization_fraction": compute_localization_fraction(
            kappa_profile,
            parameters["lags"],
            parameters["localization_radius"],
        ),
        "average_energy": average_energy,
        "relative_energy_drift": compute_relative_energy_drift(energy_history, average_energy),
    }


def append_step_metrics(histories: dict, metrics: dict) -> None:
    """Добавляет метрики шага в историю расчёта."""
    histories["time"].append(metrics["current_time"])
    histories["xi"].append(metrics["xi_profile"])
    histories["kappa"].append(metrics["kappa_profile"])
    histories["width"].append(metrics["profile_width"])
    histories["central"].append(metrics["central_amplitude"])
    histories["localization"].append(metrics["localization_fraction"])
    histories["energy"].append(metrics["average_energy"])
    histories["relative_energy_drift"].append(metrics["relative_energy_drift"])


def print_energy_drift_warning(relative_energy_drift: float, warning_threshold: float) -> None:
    """Печатает предупреждение при большом дрейфе энергии."""
    if abs(relative_energy_drift) <= warning_threshold:
        return
    print(
        "ПРЕДУПРЕЖДЕНИЕ: относительный дрейф средней энергии превысил порог: "
        f"{relative_energy_drift:.3e} > {warning_threshold:.3e}"
    )


def save_diagnostics_step(
    step_index: int,
    parameters: dict,
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    histories: dict,
) -> None:
    """Считает и сохраняет диагностики для текущего шага."""
    metrics = collect_step_metrics(
        parameters,
        step_index,
        displacement_ensemble,
        velocity_ensemble,
        histories["energy"],
    )
    append_step_metrics(histories, metrics)

    print_progress(
        step_index,
        parameters["total_steps"],
        metrics["current_time"],
        metrics["average_energy"],
        metrics["profile_width"],
        metrics["central_amplitude"],
    )
    print_energy_drift_warning(
        metrics["relative_energy_drift"],
        parameters["max_relative_energy_drift_warning"],
    )


def run_integration_loop(
    parameters: dict,
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Запускает интегрирование и периодическое сохранение диагностик."""
    histories = {
        "time": [],
        "xi": [],
        "kappa": [],
        "width": [],
        "central": [],
        "localization": [],
        "energy": [],
        "relative_energy_drift": [],
    }

    print("Запуск моделирования...")
    for step_index in range(parameters["total_steps"] + 1):
        if step_index % parameters["save_every_steps"] == 0:
            save_diagnostics_step(
                step_index,
                parameters,
                displacement_ensemble,
                velocity_ensemble,
                histories,
            )

        if step_index < parameters["total_steps"]:
            displacement_ensemble, velocity_ensemble = velocity_verlet_step(
                displacement_ensemble,
                velocity_ensemble,
                parameters["time_step"],
                parameters["stiffness"],
                parameters["beta"],
                parameters["mass"],
            )

    return displacement_ensemble, velocity_ensemble, histories


def pack_histories(histories: dict) -> dict:
    """Упаковывает историю расчёта в массивы numpy."""
    return {
        "time_grid": np.asarray(histories["time"], dtype=float),
        "xi_array": np.asarray(histories["xi"], dtype=float),
        "kappa_array": np.asarray(histories["kappa"], dtype=float),
        "width_array": np.asarray(histories["width"], dtype=float),
        "central_array": np.asarray(histories["central"], dtype=float),
        "localization_array": np.asarray(histories["localization"], dtype=float),
        "energy_array": np.asarray(histories["energy"], dtype=float),
        "relative_energy_drift_array": np.asarray(histories["relative_energy_drift"], dtype=float),
    }


def run_detector(sections: dict, parameters: dict, packed_histories: dict) -> dict:
    """Запускает детектор на сохранённой истории профилей."""
    return evaluate_breather_candidate(
        detector_settings=sections["detector"],
        time_grid=packed_histories["time_grid"],
        lags=parameters["lags"],
        kappa_history=packed_histories["kappa_array"],
        width_history=packed_histories["width_array"],
        central_amplitude_history=packed_histories["central_array"],
        localization_history=packed_histories["localization_array"],
    )


def save_results(
    config_path: Path,
    config: dict,
    sections: dict,
    output_directory: Path,
    parameters: dict,
    packed_histories: dict,
    detector_result: dict,
) -> None:
    """Сохраняет файлы с результатами расчёта."""
    save_time_series_npz(
        output_directory,
        packed_histories["time_grid"],
        parameters["lags"],
        packed_histories["xi_array"],
        packed_histories["kappa_array"],
        packed_histories["width_array"],
        packed_histories["central_array"],
        packed_histories["localization_array"],
        packed_histories["energy_array"],
        packed_histories["relative_energy_drift_array"],
    )
    save_breather_json(config_path, config, output_directory, detector_result)
    save_summary_text(output_directory, detector_result)
    save_diagnostic_plots(
        plots_enabled=bool(sections["plots"]["enabled"]),
        output_directory=output_directory,
        time_grid=packed_histories["time_grid"],
        lags=parameters["lags"],
        kappa_history=packed_histories["kappa_array"],
        width_history=packed_histories["width_array"],
        central_amplitude_history=packed_histories["central_array"],
        detector_result=detector_result,
    )


def print_final_summary(output_directory: Path, detector_result: dict) -> None:
    """Печатает итоговую сводку по завершённому расчёту."""
    print("Моделирование завершено.")
    print(f"Бризер-кандидат найден: {detector_result['found']}")
    print(f"Результаты записаны в: {output_directory}")


def run_single_experiment(config_path: Path) -> None:
    """Запускает один расчёт по файлу конфигурации."""
    config, sections, output_directory = parse_experiment_config(config_path)
    parameters = prepare_simulation_parameters(sections)
    displacement_ensemble, velocity_ensemble = create_starting_ensemble(
        parameters,
        sections["initial_conditions"],
    )
    _, _, histories = run_integration_loop(parameters, displacement_ensemble, velocity_ensemble)
    packed_histories = pack_histories(histories)
    detector_result = run_detector(sections, parameters, packed_histories)
    detector_result["final_relative_energy_drift"] = float(
        packed_histories["relative_energy_drift_array"][-1]
    )
    detector_result["max_relative_energy_drift"] = float(
        np.max(np.abs(packed_histories["relative_energy_drift_array"]))
    )
    save_results(
        config_path,
        config,
        sections,
        output_directory,
        parameters,
        packed_histories,
        detector_result,
    )
    print_final_summary(output_directory, detector_result)


def main() -> None:
    """Вызывает расчёт с конфигурацией по умолчанию."""
    run_single_experiment(Path("config.yaml"))


if __name__ == "__main__":
    main()
