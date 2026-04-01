from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def read_config(config_path: Path) -> dict:
    """Читает файл конфигурации в формате YAML."""
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_output_directory(config: dict) -> Path:
    """Создаёт каталог для выходных файлов."""
    output_directory = Path(config["experiment"]["output_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


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
    """Собирает json в новом формате без legacy-полей."""
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
        "xi_metrics": detector_result["xi_metrics"],
        "kappa_metrics": detector_result["kappa_metrics"],
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


def save_summary_text(output_directory: Path, detector_result: dict) -> None:
    """Записывает краткую текстовую сводку по результатам."""
    summary_lines = [
        f"heuristic_candidate: {detector_result['heuristic_candidate']}",
        f"timestamp: {detector_result['timestamp']}",
        "",
    ]
    summary_lines.extend(
        format_section("initial_condition_validation", detector_result["initial_condition_validation"])
    )
    summary_lines.extend(format_section("xi_metrics", detector_result["xi_metrics"]))
    summary_lines.extend(format_section("kappa_metrics", detector_result["kappa_metrics"]))
    summary_lines.extend(format_section("numerics", detector_result["numerics"]))
    summary_lines.append(f"notes: {detector_result['notes']}")

    with (output_directory / "summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(summary_lines) + "\n")
