from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def read_config(config_path: Path) -> dict:
    """Read YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_output_directory(config: dict) -> Path:
    """Create output directory if needed."""
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
    """Print clear progress line for the current saved state."""
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
    width_history: np.ndarray,
    central_amplitude_history: np.ndarray,
    localization_history: np.ndarray,
    energy_history: np.ndarray,
) -> None:
    """Save simulation time-series arrays to compressed npz file."""
    np.savez_compressed(
        output_directory / "time_series.npz",
        time_grid=time_grid,
        lags=lags,
        xi_history=xi_history,
        kappa_history=kappa_history,
        width_history=width_history,
        central_amplitude_history=central_amplitude_history,
        localization_history=localization_history,
        energy_history=energy_history,
    )


def save_breather_json(
    config_path: Path,
    config: dict,
    output_directory: Path,
    detector_result: dict,
) -> None:
    """Save detector status and main metrics in requested JSON format."""
    json_payload = {
        "found": bool(detector_result["found"]),
        "timestamp": detector_result["timestamp"],
        "config_path": str(config_path),
        "chain_length": int(config["chain"]["length"]),
        "ensemble_size": int(config["experiment"]["ensemble_size"]),
        "mass": float(config["chain"]["mass"]),
        "stiffness": float(config["chain"]["stiffness"]),
        "beta": float(config["chain"]["beta"]),
        "dt": float(config["chain"]["dt"]),
        "total_time": float(config["chain"]["dt"]) * int(config["experiment"]["total_steps"]),
        "save_every": int(config["experiment"]["save_every_steps"]),
        "detected_frequency": float(detector_result["detected_frequency"]),
        "central_amplitude": float(detector_result["central_amplitude"]),
        "final_width": float(detector_result["final_width"]),
        "oscillatory_tail_detected": bool(detector_result["oscillatory_tail_detected"]),
        "fitted_alpha": float(detector_result["fitted_alpha"]),
        "fitted_amplitude": float(detector_result["fitted_amplitude"]),
        "dominant_peak_ratio": float(detector_result["dominant_peak_ratio"]),
        "localization_score": float(detector_result["localization_score"]),
        "periodicity_score": float(detector_result["periodicity_score"]),
        "tail_fit_error": float(detector_result["tail_fit_error"]),
        "central_ratio": float(detector_result["central_ratio"]),
        "max_late_width": float(detector_result["max_late_width"]),
        "notes": str(detector_result["notes"]),
    }

    output_path = output_directory / "found_breather.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(json_payload, stream, indent=2, ensure_ascii=False)


def save_summary_text(output_directory: Path, detector_result: dict) -> None:
    """Write concise human-readable summary text file."""
    summary_lines = [
        f"found: {detector_result['found']}",
        f"timestamp: {detector_result['timestamp']}",
        f"detected_frequency: {detector_result['detected_frequency']:.6f}",
        f"final_width: {detector_result['final_width']:.6f}",
        f"central_amplitude: {detector_result['central_amplitude']:.6f}",
        f"dominant_peak_ratio: {detector_result['dominant_peak_ratio']:.6f}",
        f"tail_fit_error: {detector_result['tail_fit_error']:.6f}",
        f"notes: {detector_result['notes']}",
    ]

    with (output_directory / "summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(summary_lines) + "\n")
