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


def run_single_experiment(config_path: Path) -> None:
    """Run one experiment from a single YAML config file."""
    config = read_config(config_path)
    experiment_config = config["experiment"]
    chain_config = config["chain"]
    initial_conditions_config = config["initial_conditions"]
    correlations_config = config["correlations"]

    output_directory = ensure_output_directory(config)
    random_seed = int(experiment_config["random_seed"])
    np.random.seed(random_seed)

    ensemble_size = int(experiment_config["ensemble_size"])
    chain_length = int(chain_config["length"])
    mass = float(chain_config["mass"])
    stiffness = float(chain_config["stiffness"])
    beta = float(chain_config["beta"])
    time_step = float(chain_config["dt"])

    displacement_ensemble, velocity_ensemble = create_initial_ensemble(
        ensemble_size=ensemble_size,
        chain_length=chain_length,
        mode=str(initial_conditions_config["mode"]),
        zero_displacements=bool(initial_conditions_config["zero_displacements"]),
        random_thermal_std=float(initial_conditions_config["random_thermal_std"]),
        amplitude=float(initial_conditions_config["amplitude"]),
        alpha=float(initial_conditions_config["alpha"]),
        covariance_psd_tolerance=float(initial_conditions_config["covariance_psd_tolerance"]),
        covariance_negative_warning_tolerance=float(
            initial_conditions_config["covariance_negative_warning_tolerance"]
        ),
    )

    max_lag = int(correlations_config["max_lag"])
    localization_radius = int(correlations_config["localization_radius"])
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    total_steps = int(experiment_config["total_steps"])
    save_every_steps = int(experiment_config["save_every_steps"])

    time_history: list[float] = []
    xi_history: list[np.ndarray] = []
    kappa_history: list[np.ndarray] = []
    width_history: list[float] = []
    central_amplitude_history: list[float] = []
    localization_history: list[float] = []
    energy_history: list[float] = []

    print("Starting simulation...")
    for step_index in range(total_steps + 1):
        if step_index % save_every_steps == 0:
            current_time = step_index * time_step
            xi_profile, kappa_profile = compute_correlation_profiles(
                displacement_ensemble,
                velocity_ensemble,
                lags,
            )
            profile_width = compute_correlation_width(kappa_profile, lags)
            central_index = max_lag
            central_amplitude = float(kappa_profile[central_index])
            localization_fraction = compute_localization_fraction(
                kappa_profile,
                lags,
                localization_radius,
            )
            average_energy = compute_total_energy_ensemble(
                displacement_ensemble,
                velocity_ensemble,
                mass,
                stiffness,
                beta,
            )

            time_history.append(current_time)
            xi_history.append(xi_profile)
            kappa_history.append(kappa_profile)
            width_history.append(profile_width)
            central_amplitude_history.append(central_amplitude)
            localization_history.append(localization_fraction)
            energy_history.append(average_energy)

            print_progress(
                step_index,
                total_steps,
                current_time,
                average_energy,
                profile_width,
                central_amplitude,
            )

        if step_index < total_steps:
            displacement_ensemble, velocity_ensemble = velocity_verlet_step(
                displacement_ensemble,
                velocity_ensemble,
                time_step,
                stiffness,
                beta,
                mass,
            )

    time_grid = np.asarray(time_history, dtype=float)
    xi_array = np.asarray(xi_history, dtype=float)
    kappa_array = np.asarray(kappa_history, dtype=float)
    width_array = np.asarray(width_history, dtype=float)
    central_array = np.asarray(central_amplitude_history, dtype=float)
    localization_array = np.asarray(localization_history, dtype=float)
    energy_array = np.asarray(energy_history, dtype=float)

    detector_result = evaluate_breather_candidate(
        config=config,
        time_grid=time_grid,
        lags=lags,
        kappa_history=kappa_array,
        width_history=width_array,
        central_amplitude_history=central_array,
        localization_history=localization_array,
    )

    save_time_series_npz(
        output_directory,
        time_grid,
        lags,
        xi_array,
        kappa_array,
        width_array,
        central_array,
        localization_array,
        energy_array,
    )
    save_breather_json(config_path, config, output_directory, detector_result)
    save_summary_text(output_directory, detector_result)
    save_diagnostic_plots(
        config,
        output_directory,
        time_grid,
        lags,
        kappa_array,
        width_array,
        central_array,
        detector_result,
    )

    print("Simulation completed.")
    print(f"Breather candidate found: {detector_result['found']}")
    print(f"Results written to: {output_directory}")


if __name__ == "__main__":
    default_config_path = Path("config.yaml")
    run_single_experiment(default_config_path)
