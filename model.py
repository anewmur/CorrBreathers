from __future__ import annotations

import numpy as np


def compute_strain(displacement_ensemble: np.ndarray) -> np.ndarray:
    """Compute strain eps_n = u_{n+1} - u_n with periodic boundaries."""
    return np.roll(displacement_ensemble, shift=-1, axis=1) - displacement_ensemble


def compute_acceleration(
    displacement_ensemble: np.ndarray,
    stiffness: float,
    beta: float,
    mass: float,
) -> np.ndarray:
    """Compute accelerations for all ensemble states in the beta-FPU chain."""
    strain = compute_strain(displacement_ensemble)
    linear_force = strain - np.roll(strain, shift=1, axis=1)
    nonlinear_force = strain ** 3 - np.roll(strain ** 3, shift=1, axis=1)
    acceleration = (stiffness / mass) * (linear_force + beta * nonlinear_force)
    return acceleration


def compute_total_energy_ensemble(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    mass: float,
    stiffness: float,
    beta: float,
) -> float:
    """Compute average total energy per realization for stability diagnostics."""
    strain = compute_strain(displacement_ensemble)
    kinetic_energy = 0.5 * mass * np.sum(velocity_ensemble ** 2, axis=1)
    linear_potential = 0.5 * stiffness * np.sum(strain ** 2, axis=1)
    nonlinear_potential = 0.25 * beta * stiffness * np.sum(strain ** 4, axis=1)
    total_energy = kinetic_energy + linear_potential + nonlinear_potential
    return float(np.mean(total_energy))
