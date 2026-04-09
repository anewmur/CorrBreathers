from __future__ import annotations

import numpy as np


def create_initial_ensemble(
    ensemble_size: int,
    chain_length: int,
    mode: str,
    zero_displacements: bool,
    remove_center_of_mass_velocity: bool,
    random_thermal_std: float = 1.0,
    amplitude: float = 1.0,
    alpha: float = 0.25,
    covariance_psd_tolerance: float = 1.0e-10,
    covariance_negative_warning_tolerance: float = 1.0e-8,
    custom_nonnegative_profile: list[float] | None = None,
    # Параметры для localized_custom_covariance
    localization_center: int = 0,
    localization_half_width: int = 10,
    correlated_velocity_template: str | None = None,
    localized_covariance_profile: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт начальные смещения и скорости для ансамбля."""
    if zero_displacements:
        displacement_ensemble = np.zeros((ensemble_size, chain_length), dtype=float)
    else:
        displacement_ensemble = np.random.normal(0.0, 1.0, size=(ensemble_size, chain_length))

    if mode == "random_thermal":
        velocity_ensemble = _sample_random_thermal_velocities(
            ensemble_size, chain_length, random_thermal_std,
        )

    elif mode == "correlated_velocity":
        velocity_ensemble = _sample_correlated_velocities_fft(
            ensemble_size, chain_length, amplitude, alpha,
            covariance_psd_tolerance, covariance_negative_warning_tolerance,
        )

    elif mode == "custom_covariance_profile":
        if custom_nonnegative_profile is None:
            raise ValueError(
                "Для custom_covariance_profile нужен custom_nonnegative_profile."
            )
        velocity_ensemble = _sample_custom_correlated_velocities_fft(
            ensemble_size, chain_length, custom_nonnegative_profile,
            covariance_psd_tolerance, covariance_negative_warning_tolerance,
        )

    elif mode == "localized_custom_covariance":
        profile = _build_localized_profile(
            localized_covariance_profile=localized_covariance_profile,
            correlated_velocity_template=correlated_velocity_template,
            localization_half_width=localization_half_width,
            amplitude=amplitude,
            alpha=alpha,
        )
        velocity_ensemble = _sample_localized_correlated_velocities(
            ensemble_size=ensemble_size,
            chain_length=chain_length,
            profile=profile,
            center_index=localization_center,
        )

    else:
        raise ValueError(f"Неизвестный initial_conditions.mode: {mode}")

    if remove_center_of_mass_velocity:
        velocity_ensemble -= np.mean(velocity_ensemble, axis=1, keepdims=True)
        print(
            "ПРЕДУПРЕЖДЕНИЕ: вычтена средняя скорость по узлам; "
            "ковариация может не совпадать с заданной."
        )

    return displacement_ensemble, velocity_ensemble


# ---------------------------------------------------------------------------
#   Построение локализованного профиля
# ---------------------------------------------------------------------------

def _build_localized_profile(
    localized_covariance_profile: list[float] | None,
    correlated_velocity_template: str | None,
    localization_half_width: int,
    amplitude: float,
    alpha: float,
) -> np.ndarray:
    """Строит полный профиль [-W, ..., 0, ..., W] для localized_custom_covariance."""
    if localized_covariance_profile is not None:
        profile = np.asarray(localized_covariance_profile, dtype=float)
        if len(profile) % 2 == 0:
            raise ValueError(
                "Длина localized_covariance_profile должна быть нечётной (2W+1)."
            )
        return profile

    if correlated_velocity_template == "exponential":
        W = localization_half_width
        nonneg = np.array([
            amplitude * ((-1.0) ** k) * np.exp(-alpha * k)
            for k in range(W + 1)
        ])
        # nonneg = [C(0), C(1), ..., C(W)]
        # полный профиль: [C(W), ..., C(1), C(0), C(1), ..., C(W)]
        return np.concatenate([nonneg[:0:-1], nonneg])

    raise ValueError(
        "Для localized_custom_covariance укажите localized_covariance_profile "
        "или correlated_velocity_template='exponential'."
    )


# ---------------------------------------------------------------------------
#   Генерация локализованных скоростей через Холецкого
# ---------------------------------------------------------------------------

def _build_localized_covariance_matrix(profile: np.ndarray) -> np.ndarray:
    """Строит матрицу ковариации из профиля [-W, ..., W].

    Элемент C[i,j] = profile[W + (j-i)] если |j-i| <= W, иначе 0.
    """
    size = len(profile)
    W = (size - 1) // 2
    C = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            lag = j - i
            if abs(lag) <= W:
                C[i, j] = profile[W + lag]
    return C


def _sample_localized_correlated_velocities(
    ensemble_size: int,
    chain_length: int,
    profile: np.ndarray,
    center_index: int,
) -> np.ndarray:
    """Генерирует скорости с заданной ковариацией в окне; вне окна — нули."""
    W = (len(profile) - 1) // 2
    window_size = 2 * W + 1

    C = _build_localized_covariance_matrix(profile)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1.0e-12)
        C_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(C_fixed)
        print(
            "ПРЕДУПРЕЖДЕНИЕ: матрица ковариации не положительно определена; "
            "отрицательные собственные значения обрезаны."
        )

    window_indices = np.array(
        [(int(center_index) - W + offset) % chain_length for offset in range(window_size)],
        dtype=int,
    )

    white_noise = np.random.normal(0.0, 1.0, size=(ensemble_size, window_size))
    correlated_window = white_noise @ L.T

    velocities = np.zeros((ensemble_size, chain_length), dtype=float)
    velocities[:, window_indices] = correlated_window
    return velocities


# ---------------------------------------------------------------------------
#   Валидация начальной ковариации
# ---------------------------------------------------------------------------

def validate_initial_covariance(
    velocity_ensemble: np.ndarray,
    lags: np.ndarray,
    target_profile: np.ndarray,
) -> dict:
    """Сравнивает эмпирический профиль kappa_k(0) с целевым профилем."""
    empirical_values: list[float] = []

    for lag in lags:
        shifted_velocity = np.roll(velocity_ensemble, shift=-int(lag), axis=1)
        empirical_values.append(float(np.mean(velocity_ensemble * shifted_velocity)))

    empirical_profile = np.asarray(empirical_values, dtype=float)
    target_profile_array = np.asarray(target_profile, dtype=float)

    max_abs_error = float(np.max(np.abs(empirical_profile - target_profile_array)))

    target_norm = float(np.linalg.norm(target_profile_array))
    if target_norm <= 1.0e-14:
        relative_error = 0.0
    else:
        relative_error = float(
            np.linalg.norm(empirical_profile - target_profile_array) / target_norm
        )

    return {
        "empirical_profile": empirical_profile,
        "target_profile": target_profile_array,
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
    }


# ---------------------------------------------------------------------------
#   Однородные генераторы скоростей (без изменений)
# ---------------------------------------------------------------------------

def _sample_random_thermal_velocities(
    ensemble_size: int,
    chain_length: int,
    thermal_std: float,
) -> np.ndarray:
    """Генерирует независимые гауссовы скорости."""
    return np.random.normal(0.0, thermal_std, size=(ensemble_size, chain_length))


def _build_circulant_profile_from_nonnegative_lags(
    chain_length: int,
    nonnegative_profile: list[float],
) -> np.ndarray:
    """Строит первый ряд циркулянтной ковариации по значениям для лагов 0, 1, 2, ..."""
    correlation_values = np.zeros(chain_length, dtype=float)
    max_defined_lag = min(len(nonnegative_profile) - 1, chain_length // 2)

    for lag_index in range(max_defined_lag + 1):
        value = float(nonnegative_profile[lag_index])
        correlation_values[lag_index] = value
        if lag_index != 0:
            correlation_values[-lag_index] = value

    return correlation_values


def _sample_custom_correlated_velocities_fft(
    ensemble_size: int,
    chain_length: int,
    nonnegative_profile: list[float],
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
) -> np.ndarray:
    """Генерирует скорости по явно заданному профилю kappa_k для неотрицательных лагов."""
    correlation_values = _build_circulant_profile_from_nonnegative_lags(
        chain_length=chain_length,
        nonnegative_profile=nonnegative_profile,
    )

    clipped_spectrum = _validate_and_prepare_spectrum(
        correlation_values=correlation_values,
        covariance_psd_tolerance=covariance_psd_tolerance,
        covariance_negative_warning_tolerance=covariance_negative_warning_tolerance,
    )
    filter_amplitudes = np.sqrt(np.maximum(clipped_spectrum, 0.0))

    white_noise = np.random.normal(0.0, 1.0, size=(ensemble_size, chain_length))
    white_noise_spectrum = np.fft.fft(white_noise, axis=1)
    shaped_spectrum = white_noise_spectrum * filter_amplitudes[None, :]
    return np.real(np.fft.ifft(shaped_spectrum, axis=1))


def _sample_correlated_velocities_fft(
    ensemble_size: int,
    chain_length: int,
    amplitude: float,
    alpha: float,
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
) -> np.ndarray:
    """Генерирует скорости через БПФ с циркулянтной ковариацией."""
    lag_index = np.arange(chain_length, dtype=float)
    lag_index = np.minimum(lag_index, chain_length - lag_index)
    correlation_values = amplitude * ((-1.0) ** lag_index) * np.exp(-alpha * lag_index)

    clipped_spectrum = _validate_and_prepare_spectrum(
        correlation_values,
        covariance_psd_tolerance,
        covariance_negative_warning_tolerance,
    )
    filter_amplitudes = np.sqrt(np.maximum(clipped_spectrum, 0.0))

    white_noise = np.random.normal(0.0, 1.0, size=(ensemble_size, chain_length))
    white_noise_spectrum = np.fft.fft(white_noise, axis=1)
    shaped_spectrum = white_noise_spectrum * filter_amplitudes[None, :]
    return np.real(np.fft.ifft(shaped_spectrum, axis=1))


def _validate_and_prepare_spectrum(
    correlation_values: np.ndarray,
    covariance_psd_tolerance: float,
    covariance_negative_warning_tolerance: float,
) -> np.ndarray:
    """Проверяет спектр плотности и обрезает малые отрицательные значения."""
    spectrum = np.real(np.fft.fft(correlation_values))
    minimum_spectrum_value = float(np.min(spectrum))

    if minimum_spectrum_value >= -covariance_psd_tolerance:
        return np.clip(spectrum, a_min=0.0, a_max=None)

    if minimum_spectrum_value >= -covariance_negative_warning_tolerance:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: в спектре ковариации есть малые отрицательные значения; "
            "они обрезаны до нуля."
        )
        return np.clip(spectrum, a_min=0.0, a_max=None)

    raise ValueError(
        "Спектральная плотность ковариации заметно отрицательна. "
        f"Минимум спектра: {minimum_spectrum_value:.3e}."
    )
