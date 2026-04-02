from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.animation import FuncAnimation


def show_kappa_profile_animation(
    npz_path: str | Path,
    interval_ms: int = 80,
) -> None:
    """Показывает анимацию эволюции профиля kappa_k(t) в окне matplotlib."""
    npz_path = Path(npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Файл не найден: {npz_path}")

    npz_file = np.load(npz_path)
    time_grid = npz_file["time_grid"]
    lags = npz_file["lags"]
    kappa_history = npz_file["kappa_history"]

    frame_count = int(kappa_history.shape[0])
    if frame_count == 0:
        raise ValueError("В time_series.npz нет кадров для анимации.")

    figure, axes = plt.subplots(figsize=(8, 4.5))

    y_min = float(np.min(kappa_history))
    y_max = float(np.max(kappa_history))
    y_margin = 0.05 * max(y_max - y_min, 1.0e-12)

    axes.set_xlim(float(np.min(lags)), float(np.max(lags)))
    axes.set_ylim(y_min - y_margin, y_max + y_margin)
    axes.set_xlabel("лаг k")
    axes.set_ylabel("kappa_k")
    axes.set_title("Эволюция профиля kappa_k(t)")
    axes.grid(True, alpha=0.3)

    line, = axes.plot([], [], marker="o", markersize=3, linewidth=1.5)
    time_text = axes.text(
        0.02,
        0.95,
        "",
        transform=axes.transAxes,
        ha="left",
        va="top",
    )

    def init() -> tuple:
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(frame_index: int) -> tuple:
        profile = kappa_history[frame_index]
        current_time = float(time_grid[frame_index])

        line.set_data(lags, profile)
        time_text.set_text(f"t = {current_time:.3f}")
        return line, time_text

    animation = FuncAnimation(
        figure,
        update,
        init_func=init,
        frames=frame_count,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    show_kappa_profile_animation(
        Path("results") / "T100_beta02_ens200_ini-cov_001" / "time_series.npz"
    )