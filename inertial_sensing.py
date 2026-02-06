"""
This script plots allan-deviation of our IMU data (x,y,z axes of both accelerometer and gyroscope).
We're using Allan variance to determine noise parameters of a MEMS gyroscope and how noise behaves 
as you average longer and longer. We also plotted additional breakpoints for white noise, bias instability, and random walk.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "imu_allan_1hour15min.csv"
FS_HZ = 100.0
TARGET_COLUMN = "accel_z"


def read_columns(csv_path, target_columns):
    """Read multiple columns from CSV. Returns dict of column_name -> 1D array."""
    out = {c: [] for c in target_columns}
    header = None
    indices = None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            if header is None:
                header = [h.strip() for h in row]
                missing = [c for c in target_columns if c not in header]
                if missing:
                    raise ValueError(f"Columns not found: {missing}. Header: {header}")
                indices = {c: header.index(c) for c in target_columns}
                continue

            try:
                for c in target_columns:
                    out[c].append(float(row[indices[c]]))
            except (IndexError, ValueError):
                continue

    for c in target_columns:
        out[c] = np.asarray(out[c], dtype=float)
        if out[c].size < 1000:
            raise ValueError(f"Too few samples for '{c}' ({out[c].size}). Check parsing.")
    return out


def m_grid(N, num=40):
    max_m = 2 ** int(np.floor(np.log2(N // 2)))
    m = np.logspace(0, np.log10(max_m), num=num)
    m = np.unique(np.ceil(m).astype(int))
    m = m[m >= 1]
    return m

def draw_slope_line(ax, tau0, sigma0, slope, label, label_frac=0.5, va="center", ha="left", **text_kw):
    """
    Draw a reference line with given slope through (tau0, sigma0)
    on a log-log Allan deviation plot.
    label_frac: 0=start, 0.5=middle, 1=end. va/ha: alignment.
    """
    t = np.array([tau0 / 5, tau0 * 5])
    s = sigma0 * (t / tau0) ** slope
    ax.loglog(t, s, "--", linewidth=2)
    t_txt = t[0] * (t[1] / t[0]) ** label_frac
    s_txt = sigma0 * (t_txt / tau0) ** slope
    kw = {"fontsize": 10, **text_kw}
    ax.text(t_txt, s_txt, label, va=va, ha=ha, **kw)


"""
From the MATLAB documentation (https://www.mathworks.com/help/fusion/ug/inertial-sensor-noise-analysis-using-allan-variance.html)
There are two ways of computing the allan deviation, based on accelerometer or gyroscope.

dev rate is for accelerometer data:
    Uses cluster-mean difference form:
        sigma^2(tau) = 0.5 * mean( (xbar[k+1] - xbar[k])^2 )
    where xbar are averages over window length tau.

theta rate is for gyroscope data:
    Allan deviation using the theta second-difference form (gyro standard):
    theta = cumsum(omega) * t0
    sigma^2(tau) = sum( (theta[k+2m]-2theta[k+m]+theta[k])^2 ) / (2*tau^2*(L-2m))
"""
def allan_dev_rate(x, fs):
    x = np.asarray(x, dtype=float)
    N = x.size
    t0 = 1.0 / fs
    m_vals = m_grid(N)

    taus = []
    adev = []

    for m in m_vals:
        if 2 * m >= N:
            break
        K = N // m
        x_trim = x[:K * m]
        x_bar = np.mean(x_trim.reshape(K, m), axis=1)
        diff = np.diff(x_bar)
        sigma2 = 0.5 * np.mean(diff ** 2)
        taus.append(m * t0)
        adev.append(np.sqrt(sigma2))

    return np.asarray(taus), np.asarray(adev)


def allan_dev_theta(omega, fs):
    omega = np.asarray(omega, dtype=float)
    t0 = 1.0 / fs
    theta = np.cumsum(omega) * t0
    L = theta.size

    m_vals = m_grid(L)
    tau = m_vals * t0
    avar = np.zeros_like(tau, dtype=float)

    valid = 0
    for i, m in enumerate(m_vals):
        if 2 * m >= L:
            break
        d = theta[2*m:] - 2 * theta[m:-m] + theta[:-2*m]
        avar[i] = np.sum(d ** 2) / (2.0 * (tau[i] ** 2) * (L - 2*m))
        valid += 1

    adev = np.sqrt(avar[:valid])
    return tau[:valid], adev

data = read_columns(CSV_PATH, [TARGET_COLUMN])
x = data[TARGET_COLUMN]


if TARGET_COLUMN.startswith("gyro"):
    taus, adev = allan_dev_theta(x, FS_HZ)
else:
    taus, adev = allan_dev_rate(x, FS_HZ)

slope = np.gradient(np.log10(adev), np.log10(taus))
white_noise_idx = np.argmin(np.abs(slope + 0.5))
white_noise_tau = taus[white_noise_idx]
bias_idx = np.argmin(adev)
bias_tau = taus[bias_idx]
random_walk_idx = np.argmin(np.abs(slope - 0.5))
random_walk_tau = taus[random_walk_idx]



fig, ax = plt.subplots(figsize=(15, 8))
ax.loglog(taus, adev, linewidth=3)
ax.set_ylim(1e-1,1e4)
ax.set_xlabel(r"$\tau$ [s]")
ax.set_ylabel("Allan Deviation")
ax.set_title(f"Allan Deviation: {TARGET_COLUMN}")
ax.grid(True, which="both")



draw_slope_line(ax, white_noise_tau, adev[white_noise_idx], slope=-0.5,
    label="White noise (slope −1/2)", label_frac=0.2, va="bottom")
draw_slope_line(ax, bias_tau, adev[bias_idx], slope=0.0,
    label="Bias instability (slope 0)", label_frac=0.5, va="top")
draw_slope_line(ax, random_walk_tau, adev[random_walk_idx], slope=+0.5,
    label="Random walk (slope +1/2)", label_frac=0.8, va="bottom")

plt.tight_layout()
plt.show()