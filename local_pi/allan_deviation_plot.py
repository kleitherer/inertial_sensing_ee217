"""
This script plots allan-deviation of our IMU data (x,y,z axes of both accelerometer and gyroscope).

 1hour 15 minutes long csv of IMU data.

we're using Allan variance to determine noise parameters of a MEMS gyroscope, and how
noise behaves as you average longer and longer

These parameters can be used to model the gyroscope in simulation.

obtain the averages of the sum of the data points contained in each cluster over the length of the cluster. 

"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

CSV_PATH = 'imu_allan_1hour15min.csv'
TARGET_COLUMN = "gyro_x_raw"
"""
# fs_hz,100.0
# duration_s,4500
# gyro_offset_counts_x,-4
# gyro_offset_counts_y,-32
# gyro_offset_counts_z,-4

t_s,accel_x,accel_y,accel_z,gyro_x_raw,gyro_y_raw,gyro_z_raw,gyro_x_corr,gyro_y_corr,gyro_z_corr
"""

def read_column(csv_path, target_column):
    data = []
    header = None
    col_idx = None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # First non-comment line should be header
            if header is None:
                header = [h.strip() for h in row]
                if target_column not in header:
                    raise ValueError(f"Column '{target_column}' not found. Header: {header}")
                col_idx = header.index(target_column)
                continue

            # Data rows
            try:
                data.append(float(row[col_idx]))
            except (IndexError, ValueError):
                # skip malformed lines
                continue

    if len(data) < 100:
        raise ValueError(f"Too few samples read ({len(data)}). Check file/column/header parsing.")
    return np.array(data, dtype=float)

def allan_deviation(x, fs_hz):
    x = np.asarray(x, dtype=float)
    N = len(x)
    T0 = 1.0 / fs_hz

    max_m = N // 10
    if max_m < 2:
        raise ValueError("Not enough data for Allan deviation (need N >= ~20).")

    m_vals = np.unique(np.logspace(0, np.log10(max_m), num=40).astype(int))

    taus = []
    adev = []

    for m in m_vals:
        if 2 * m >= N:
            break

        K = N // m
        x_trim = x[:K * m]
        x_bar = np.mean(x_trim.reshape(K, m), axis=1)

        diff = np.diff(x_bar)
        sigma2 = 0.5 * np.mean(diff**2)

        taus.append(m * T0)
        adev.append(np.sqrt(sigma2))

    return np.array(taus), np.array(adev)

x = read_column(CSV_PATH, TARGET_COLUMN)
taus, adev = allan_deviation(x, 100.0)

plt.figure()
plt.loglog(taus, adev)
plt.xlabel(r'$\tau$ [s]')
plt.ylabel('Allan Deviation [counts]')
plt.title(f'Allan Deviation: {TARGET_COLUMN} (fs={100.0} Hz)')
plt.grid(True, which='both')
plt.show()
# specify which sensor to plot
# specify which axis to plot
# on x-axis, plot the log scale of tao which is the cluster of samples, where m * T0 is averaging window

# compute cluster average
# compute allan variance
# compute allan deviation, which becomes the y-axis