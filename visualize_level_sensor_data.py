"""
Same as visualize_level.py but explicitly uses sensor_data.csv.
Columns: timestamp, accel_x_mps2, accel_y_mps2, accel_z_mps2, gyro_x_dps, gyro_y_dps, gyro_z_dps.

Plots: (1) accel x/y/z, (2) gyro x/y/z, (3) angular deviation from level (accel vs gyro).
"""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "sensor_data.csv"


def load_sensor_data(csv_path):
    """Load sensor data from CSV. Returns timestamps, accel x/y/z (m/s²), gyro x/y/z (DPS)."""
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, csv_path)

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return None

    with open(csv_file, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            timestamps.append(datetime.fromisoformat(row["timestamp"]))
            accel_x.append(float(row["accel_x_mps2"]))
            accel_y.append(float(row["accel_y_mps2"]))
            accel_z.append(float(row["accel_z_mps2"]))
            gyro_x.append(float(row["gyro_x_dps"]))
            gyro_y.append(float(row["gyro_y_dps"]))
            gyro_z.append(float(row["gyro_z_dps"]))

    print(f"Loaded {len(timestamps)} points from {csv_file}")
    return timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z


def plot_sensor_data(timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
    ax = np.array(accel_x)
    ay = np.array(accel_y)
    az = np.array(accel_z)
    gx, gy = np.array(gyro_x), np.array(gyro_y)

    t0 = timestamps[0]
    time_s = np.array([(t - t0).total_seconds() for t in timestamps])
    dt = np.diff(time_s)

    roll_accel = np.degrees(np.arctan2(ay, az))
    pitch_accel = np.degrees(np.arctan2(-ax, az))
    gyro_roll = np.concatenate([[0], np.cumsum(gx[:-1] * dt)])
    gyro_pitch = np.concatenate([[0], np.cumsum(gy[:-1] * dt)])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("ICM-20948 Sensor Data: sensor_data.csv", fontsize=16, fontweight="bold")

    ax1.plot(time_s, accel_x, "r-", label="Accel X (m/s²)", lw=1.5)
    ax1.plot(time_s, accel_y, "g-", label="Accel Y (m/s²)", lw=1.5)
    ax1.plot(time_s, accel_z, "b-", label="Accel Z (m/s²)", lw=1.5)
    ax1.set_title("Accelerometer (m/s²)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_s, gyro_x, "r--", label="Gyro X (DPS)", lw=1.5, alpha=0.8)
    ax2.plot(time_s, gyro_y, "g--", label="Gyro Y (DPS)", lw=1.5, alpha=0.8)
    ax2.plot(time_s, gyro_z, "b--", label="Gyro Z (DPS)", lw=1.5, alpha=0.8)
    ax2.set_title("Gyroscope (DPS)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular velocity (DPS)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time_s, roll_accel, "r-", label="Roll (accel): atan2(ay, az)", lw=1.5, alpha=0.8)
    ax3.plot(time_s, pitch_accel, "g-", label="Pitch (accel): atan2(-ax, az)", lw=1.5, alpha=0.8)
    ax3.plot(time_s, gyro_roll, "r--", label="Roll (gyro): ∫gyro_x dt", lw=1.5, alpha=0.8)
    ax3.plot(time_s, gyro_pitch, "g--", label="Pitch (gyro): ∫gyro_y dt", lw=1.5, alpha=0.8)
    ax3.axhline(0, color="k", ls=":", lw=0.5)
    ax3.set_title("Angular deviation from level (deg) — Accel vs Gyro")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (deg)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Sensor Data Statistics")
    print("=" * 60)
    print("\nAccelerometer (m/s²):")
    print(f"  X: min={min(accel_x):.3f}, max={max(accel_x):.3f}, mean={np.mean(accel_x):.3f}, std={np.std(accel_x):.3f}")
    print(f"  Y: min={min(accel_y):.3f}, max={max(accel_y):.3f}, mean={np.mean(accel_y):.3f}, std={np.std(accel_y):.3f}")
    print(f"  Z: min={min(accel_z):.3f}, max={max(accel_z):.3f}, mean={np.mean(accel_z):.3f}, std={np.std(accel_z):.3f}")
    print("\nGyroscope (DPS):")
    print(f"  X: min={min(gyro_x):.3f}, max={max(gyro_x):.3f}, mean={np.mean(gyro_x):.3f}, std={np.std(gyro_x):.3f}")
    print(f"  Y: min={min(gyro_y):.3f}, max={max(gyro_y):.3f}, mean={np.mean(gyro_y):.3f}, std={np.std(gyro_y):.3f}")
    print(f"  Z: min={min(gyro_z):.3f}, max={max(gyro_z):.3f}, mean={np.mean(gyro_z):.3f}, std={np.std(gyro_z):.3f}")
    print("=" * 60 + "\n")


def main():
    data = load_sensor_data(CSV_PATH)
    if data is None or len(data[0]) == 0:
        print("No data loaded. Exiting.")
        return
    timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = data
    plot_sensor_data(timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)


if __name__ == "__main__":
    main()
