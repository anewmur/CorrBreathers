from __future__ import annotations

import numpy as np

from model import compute_acceleration


def velocity_verlet_step(
    displacement_ensemble: np.ndarray,
    velocity_ensemble: np.ndarray,
    time_step: float,
    stiffness: float,
    beta: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Делает один шаг схемы Верле для всего ансамбля."""
    old_acceleration = compute_acceleration(
        displacement_ensemble,
        stiffness,
        beta,
        mass,
    )
    next_displacement = (
        displacement_ensemble
        + time_step * velocity_ensemble
        + 0.5 * (time_step ** 2) * old_acceleration
    )

    next_acceleration = compute_acceleration(
        next_displacement,
        stiffness,
        beta,
        mass,
    )
    next_velocity = velocity_ensemble + 0.5 * time_step * (old_acceleration + next_acceleration)

    return next_displacement, next_velocity
