"""
This script provides an output plot of the angular deviation from level (0 degrees) based on the csv data we collected
as we tilted our raspberryPi + accelerometer/gyroscope hat at the following (ground truth) angles:
- 

"""

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'csv_data2.csv'

df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
t_start = df['timestamp'].iloc[0]

df['time_since_start_s'] = (df['timestamp'] - t_start).dt.total_seconds()

fig, ax = plt.subplots()
ax.plot(df['time_since_start_s'], df['roll_accel_deg'], label='roll_accel_deg', alpha=0.8)
ax.plot(df['time_since_start_s'], df['gyro_y_angle_deg'], label='gyro_y_angle_deg', alpha=0.8)
ax.set_xlabel('Time since start (s)')
ax.set_ylabel('Angle (°)')
ax.set_title('Angular deviation from level (0°)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()