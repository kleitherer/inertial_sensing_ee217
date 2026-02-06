#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Part One: 

Write Python code that implements a simple gravity level; your code should plot, in real time, the angular
deviation from level, where level is at 0°. 

The sensing function should be implemented in two ways - one
using the gyro (pitch/roll/raw), and one with the accelerometer (x/y/z). Analyze the
performance of the two methods (gyro vs. accel) - which is superior?


This script should take sensor_data.csv and plot it over time 
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

CSV_PATH = 'sensor_data.csv'
# timestamp,accel_x_mps2,accel_y_mps2,accel_z_mps2,gyro_x_dps,gyro_y_dps,gyro_z_dps

def load_sensor_data(csv_path):
    """
    Load sensor data from CSV file
    Returns: timestamps, accelerometer data (x,y,z in m/s²), gyroscope data (x,y,z in DPS)
    """
    timestamps = []
    accel_x = []
    accel_y = []
    accel_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, csv_path)
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return None
    
    try:
        with open(csv_file, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                # Parse timestamp
                timestamp = datetime.fromisoformat(row['timestamp'])
                timestamps.append(timestamp)
                
                # Parse accelerometer data (m/s²)
                accel_x.append(float(row['accel_x_mps2']))
                accel_y.append(float(row['accel_y_mps2']))
                accel_z.append(float(row['accel_z_mps2']))
                
                # Parse gyroscope data (DPS)
                gyro_x.append(float(row['gyro_x_dps']))
                gyro_y.append(float(row['gyro_y_dps']))
                gyro_z.append(float(row['gyro_z_dps']))
        
        print(f"Loaded {len(timestamps)} data points from {csv_file}")
        return timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_sensor_data(timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
    """
    Plot accelerometer and gyroscope data over time
    """
    # Convert timestamps to relative time (seconds from start)
    start_time = timestamps[0]
    time_seconds = [(t - start_time).total_seconds() for t in timestamps]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ICM-20948 Sensor Data from CSV', fontsize=16, fontweight='bold')
    
    # Plot 1: Accelerometer data (m/s²)
    ax1.plot(time_seconds, accel_x, 'r-', label='Accel X (m/s²)', linewidth=1.5)
    ax1.plot(time_seconds, accel_y, 'g-', label='Accel Y (m/s²)', linewidth=1.5)
    ax1.plot(time_seconds, accel_z, 'b-', label='Accel Z (m/s²)', linewidth=1.5)
    ax1.set_title('Accelerometer Data (m/s²)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Acceleration (m/s²)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Gyroscope data (DPS)
    ax2.plot(time_seconds, gyro_x, 'r--', label='Gyro X (DPS)', linewidth=1.5, alpha=0.8)
    ax2.plot(time_seconds, gyro_y, 'g--', label='Gyro Y (DPS)', linewidth=1.5, alpha=0.8)
    ax2.plot(time_seconds, gyro_z, 'b--', label='Gyro Z (DPS)', linewidth=1.5, alpha=0.8)
    ax2.set_title('Gyroscope Data (Degrees Per Second)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Angular Velocity (DPS)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Sensor Data Statistics:")
    print("="*60)
    print(f"\nAccelerometer (m/s²):")
    print(f"  X: min={min(accel_x):.3f}, max={max(accel_x):.3f}, mean={np.mean(accel_x):.3f}, std={np.std(accel_x):.3f}")
    print(f"  Y: min={min(accel_y):.3f}, max={max(accel_y):.3f}, mean={np.mean(accel_y):.3f}, std={np.std(accel_y):.3f}")
    print(f"  Z: min={min(accel_z):.3f}, max={max(accel_z):.3f}, mean={np.mean(accel_z):.3f}, std={np.std(accel_z):.3f}")
    print(f"\nGyroscope (DPS):")
    print(f"  X: min={min(gyro_x):.3f}, max={max(gyro_x):.3f}, mean={np.mean(gyro_x):.3f}, std={np.std(gyro_x):.3f}")
    print(f"  Y: min={min(gyro_y):.3f}, max={max(gyro_y):.3f}, mean={np.mean(gyro_y):.3f}, std={np.std(gyro_y):.3f}")
    print(f"  Z: min={min(gyro_z):.3f}, max={max(gyro_z):.3f}, mean={np.mean(gyro_z):.3f}, std={np.std(gyro_z):.3f}")
    print("="*60 + "\n")

def main():
    print("\nICM-20948 CSV Data Visualization\n")
    
    # Load data from CSV
    data = load_sensor_data(CSV_PATH)
    
    if data is None:
        print("Failed to load sensor data. Exiting.")
        return
    
    timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = data
    
    if len(timestamps) == 0:
        print("No data found in CSV file. Exiting.")
        return
    
    # Plot the data
    plot_sensor_data(timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

if __name__ == '__main__':
    main()
