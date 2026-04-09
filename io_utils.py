import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def read_config(config_path: Path) -> dict:
    """Читает файл конфигурации в формате YAML."""
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_output_directory(config: dict, config_path: Path) -> Path:
    """Создаёт отдельный каталог прогона с говорящим именем и копирует в него конфиг."""
    root_output_directory = Path(config["experiment"]["output_dir"])
    root_output_directory.mkdir(parents=True, exist_ok=True)

    run_directory_name = build_run_directory_name(config, root_output_directory)
    output_directory = root_output_directory / run_directory_name
    output_directory.mkdir(parents=True, exist_ok=False)

    copied_config_path = output_directory / "config.yaml"
    shutil.copy2(config_path, copied_config_path)

    return output_directory


def build_run_directory_name(config: dict, root_output_directory: Path) -> str:
    """Строит говорящее имя каталога прогона и добавляет порядковый номер."""
    experiment_settings = config["experiment"]
    chain_settings = config["chain"]
    initial_conditions_settings = config["initial_conditions"]

    total_steps = int(experiment_settings["total_steps"])
    ensemble_size = int(experiment_settings["ensemble_size"])
    time_step = float(chain_settings["dt"])
    beta = float(chain_settings["beta"])
    initial_mode = str(initial_conditions_settings["mode"])

    total_time_label = format_total_time_label(total_steps, time_step)
    beta_label = format_beta_label(beta)
    ensemble_label = f"ens{ensemble_size}"
    initial_mode_label = format_initial_mode_label(initial_mode)

    base_name = f"{total_time_label}_{beta_label}_{ensemble_label}_{initial_mode_label}"
    sequence_number = find_next_sequence_number(root_output_directory, base_name)

    return f"{base_name}_{sequence_number:03d}"


def format_total_time_label(total_steps: int, time_step: float) -> str:
    """Формирует метку полной длительности расчёта."""
    total_time = total_steps * time_step
    total_time_text = format_float_compact(total_time)
    return f"T{total_time_text}"


def format_beta_label(beta: float) -> str:
    """Формирует метку коэффициента beta без точки."""
    beta_text = format_float_compact(beta)
    return f"beta{beta_text}"


def format_initial_mode_label(initial_mode: str) -> str:
    """Преобразует режим начальных условий в короткую устойчивую метку."""
    mode_labels: dict[str, str] = {
        "correlated_velocity": "ini-cor",
        "random_thermal": "ini-ran",
        "custom_covariance_profile": "ini-cov",
        "localized_custom_covariance": "ini-loc",
    }
    if initial_mode not in mode_labels:
        raise ValueError(f"Неизвестный режим initial_conditions.mode: {initial_mode}")
    return mode_labels[initial_mode]


def find_next_sequence_number(root_output_directory: Path, base_name: str) -> int:
    """Возвращает следующий свободный порядковый номер для каталога прогона."""
    max_sequence_number = 0

    for child_path in root_output_directory.iterdir():
        if not child_path.is_dir():
            continue
        child_name = child_path.name
        expected_prefix = f"{base_name}_"
        if not child_name.startswith(expected_prefix):
            continue

        suffix = child_name[len(expected_prefix):]
        if not suffix.isdigit():
            continue

        sequence_number = int(suffix)
        if sequence_number > max_sequence_number:
            max_sequence_number = sequence_number

    return max_sequence_number + 1


def format_float_compact(value: float) -> str:
    """Преобразует число в компактную строку без точки и лишних нулей."""
    rounded_value = round(value, 10)

    if abs(rounded_value - round(rounded_value)) <= 1.0e-10:
        return str(int(round(rounded_value)))

    text = f"{rounded_value:.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def print_progress(
    step_index: int,
    total_steps: int,
    current_time: float,
    average_energy: float,
    profile_width: float,
    central_amplitude: float,
) -> None:
    """Печатает строку состояния на сохранённом шаге."""
    message = (
        f"step={step_index:6d}/{total_steps:6d} | "
        f"time={current_time:9.4f} | "
        f"mean_energy={average_energy:12.6f} | "
        f"width={profile_width:9.4f} | "
        f"kappa0={central_amplitude:12.6f}"
    )
    print(message)


def save_time_series_npz(
    output_directory: Path,
    time_grid: np.ndarray,
    lags: np.ndarray,
    xi_history: np.ndarray,
    kappa_history: np.ndarray,
    xi_width_history: np.ndarray,
    kappa_width_history: np.ndarray,
    xi_central_amplitude_history: np.ndarray,
    kappa_central_amplitude_history: np.ndarray,
    xi_localization_history: np.ndarray,
    kappa_localization_history: np.ndarray,
    energy_history: np.ndarray,
    relative_energy_drift_history: np.ndarray,
) -> None:
    """Сохраняет временные ряды в сжатый файл npz."""
    np.savez_compressed(
        output_directory / "time_series.npz",
        time_grid=time_grid,
        lags=lags,
        xi_history=xi_history,
        kappa_history=kappa_history,
        xi_width_history=xi_width_history,
        kappa_width_history=kappa_width_history,
        xi_central_amplitude_history=xi_central_amplitude_history,
        kappa_central_amplitude_history=kappa_central_amplitude_history,
        xi_localization_history=xi_localization_history,
        kappa_localization_history=kappa_localization_history,
        energy_history=energy_history,
        relative_energy_drift_history=relative_energy_drift_history,
    )


def build_breather_payload(config_path: Path, config: dict, detector_result: dict) -> dict:
    """Собирает json в новом формате без legacy-полей и с блоком numerics."""
    initial_validation = dict(detector_result["initial_condition_validation"])
    initial_validation["remove_center_of_mass_velocity"] = bool(
        initial_validation["remove_center_of_mass_velocity"]
    )

    return {
        "heuristic_candidate": bool(detector_result["heuristic_candidate"]),
        "timestamp": str(detector_result["timestamp"]),
        "config_path": str(config_path),
        "chain": {
            "length": int(config["chain"]["length"]),
            "mass": float(config["chain"]["mass"]),
            "stiffness": float(config["chain"]["stiffness"]),
            "beta": float(config["chain"]["beta"]),
            "dt": float(config["chain"]["dt"]),
        },
        "experiment": {
            "ensemble_size": int(config["experiment"]["ensemble_size"]),
            "total_steps": int(config["experiment"]["total_steps"]),
            "save_every_steps": int(config["experiment"]["save_every_steps"]),
            "total_time": float(config["chain"]["dt"]) * int(config["experiment"]["total_steps"]),
        },
        "initial_condition_validation": initial_validation,
        "xi_metrics": dict(detector_result["xi_metrics"]),
        "kappa_metrics": dict(detector_result["kappa_metrics"]),
        "numerics": {
            "final_relative_energy_drift": float(detector_result["numerics"]["final_relative_energy_drift"]),
            "max_relative_energy_drift": float(detector_result["numerics"]["max_relative_energy_drift"]),
        },
        "notes": str(detector_result["notes"]),
    }


def save_breather_json(
    config_path: Path,
    config: dict,
    output_directory: Path,
    detector_result: dict,
) -> None:
    """Сохраняет итоговые метрики детектора в файл json."""
    json_payload = build_breather_payload(config_path, config, detector_result)
    output_path = output_directory / "found_breather.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(json_payload, stream, indent=2, ensure_ascii=False)


def format_section(section_name: str, section_values: dict) -> list[str]:
    """Форматирует словарь как секцию summary.txt."""
    lines = [f"[{section_name}]"]
    for key, value in section_values.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    return lines


def format_initial_condition_validation_section(initial_validation: dict) -> list[str]:
    """Форматирует краткую секцию валидации начальных условий для summary.txt."""
    return [
        "[initial_condition_validation]",
        f"mode: {initial_validation['mode']}",
        f"remove_center_of_mass_velocity: {initial_validation['remove_center_of_mass_velocity']}",
        f"max_abs_error: {initial_validation['max_abs_error']}",
        f"relative_error: {initial_validation['relative_error']}",
        "",
    ]


def save_summary_text(output_directory: Path, detector_result: dict) -> None:
    """Записывает краткую текстовую сводку по результатам."""
    summary_lines = [
        f"heuristic_candidate: {detector_result['heuristic_candidate']}",
        f"timestamp: {detector_result['timestamp']}",
        "",
    ]
    summary_lines.extend(
        format_initial_condition_validation_section(detector_result["initial_condition_validation"])
    )
    summary_lines.extend(format_section("xi_metrics", detector_result["xi_metrics"]))
    summary_lines.extend(format_section("kappa_metrics", detector_result["kappa_metrics"]))
    summary_lines.extend(format_section("numerics", detector_result["numerics"]))
    summary_lines.append(f"notes: {detector_result['notes']}")

    with (output_directory / "summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(summary_lines) + "\n")