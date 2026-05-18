import json
import numpy as np

def prony_frequency(series, dt, fraction=0.2, order=2):
    n_use = max(8, int(len(series) * fraction))
    s = np.array(series[:n_use], dtype=float)
    s = s - np.mean(s)
    n = len(s)
    if n < 2 * order:
        return 0.0, 0.0

    nrows = n - order
    A = np.zeros((nrows, order))
    for j in range(order):
        A[:, j] = s[j:j + nrows]
    b = s[order:order + nrows]

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    poly = np.zeros(order + 1)
    poly[0] = 1.0
    for j in range(order):
        poly[j + 1] = -coeffs[order - 1 - j]

    roots = np.roots(poly)

    best_freq = 0.0
    best_quality = 0.0
    for root in roots:
        if np.abs(root) < 1e-14:
            continue
        freq = np.abs(np.angle(root)) / (2.0 * np.pi * dt)
        quality = np.abs(root)
        if freq > 0.01 and quality > best_quality:
            best_freq = freq
            best_quality = quality

    return best_freq, best_quality


# with open("nonlinear_shift_multi/dispersion_measurements.json", "r") as f:
#     data = json.load(f)

with open("dispersion_beta0/dispersion_measurements.json", "r") as f:
    data = json.load(f)

dt_save = 0.01 * 10  # dt * save_every_steps
f_theory_max = 2.0 / np.pi

print(f"f_theory_max = {f_theory_max:.6f}")
print()
print(f"{'amp':>6s}  {'mode':>4s}  {'f_prony_10%':>12s}  {'f_prony_20%':>12s}  {'f_prony_30%':>12s}  {'quality':>8s}  {'above?':>6s}")

for amp_key, modes_dict in data["projection_series"].items():
    amp_val = float(amp_key.replace("p", "."))
    for mode_str, series in modes_dict.items():
        mode = int(mode_str)
        if mode not in [112, 120, 125, 127]:
            continue
        f10, q10 = prony_frequency(series, dt_save, fraction=0.10)
        f20, q20 = prony_frequency(series, dt_save, fraction=0.20)
        f30, q30 = prony_frequency(series, dt_save, fraction=0.30)
        above = "YES" if f20 > f_theory_max else "no"
        print(f"{amp_val:6.2f}  {mode:4d}  {f10:12.6f}  {f20:12.6f}  {f30:12.6f}  {q20:8.4f}  {above:>6s}")