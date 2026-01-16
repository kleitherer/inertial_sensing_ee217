#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Simple example: How to read acceleration measurements from ICM-20948

This demonstrates the basic usage of reading accelerometer data.
"""

import time
import math
import sys
import os

# Add the ICM20948 module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ICM-20948', 'Raspberry Pi', 'python'))
import ICM20948

def read_acceleration(icm):
    """
    Read and convert acceleration measurements from ICM-20948
    
    Args:
        icm: ICM20948 sensor object
        
    Returns:
        accel_x, accel_y, accel_z in g's
    """
    # Read sensor data (this populates the global Accel array)
    icm.icm20948_Gyro_Accel_Read()
    
    # Access the global Accel array (contains raw LSB counts)
    accel_x_raw = ICM20948.Accel[0]
    accel_y_raw = ICM20948.Accel[1]
    accel_z_raw = ICM20948.Accel[2]
    
    # Convert to g's
    # The sensor is configured for ±2g range in the initialization
    # For ±2g range: 16384 LSB/g
    # For ±4g range: 8192 LSB/g  
    # For ±8g range: 4096 LSB/g
    # For ±16g range: 2048 LSB/g
    ACCEL_SCALE = 16384.0  # Current configuration is ±2g
    
    accel_x_g = accel_x_raw / ACCEL_SCALE
    accel_y_g = accel_y_raw / ACCEL_SCALE
    accel_z_g = accel_z_raw / ACCEL_SCALE
    
    return accel_x_g, accel_y_g, accel_z_g


if __name__ == '__main__':
    print("\n=== Reading Acceleration from ICM-20948 ===\n")
    
    try:
        # Initialize sensor
        icm = ICM20948.ICM20948()
        print("Sensor initialized successfully\n")
        print("Reading acceleration measurements...")
        print("Press Ctrl+C to stop\n")
        print("=" * 70)
        
        while True:
            # Read acceleration
            accel_x, accel_y, accel_z = read_acceleration(icm)
            
            # Compute magnitude
            accel_magnitude = (accel_x**2 + accel_y**2 + accel_z**2)**0.5
            
            # When stationary, this should be close to 1g (gravity)
            print(f"Acceleration: X={accel_x:7.3f}g, Y={accel_y:7.3f}g, Z={accel_z:7.3f}g")
            print(f"Magnitude: {accel_magnitude:.3f}g (should be ~1.0g when stationary)")
            
            # Compute pitch and roll from accelerometer (when not moving)
            pitch_rad = -math.asin(accel_x / accel_magnitude)
            roll_rad = math.atan2(accel_y, accel_z)
            pitch_deg = math.degrees(pitch_rad)
            roll_deg = math.degrees(roll_rad)
            
            print(f"Pitch (from accel): {pitch_deg:7.2f}°, Roll: {roll_deg:7.2f}°")
            print("-" * 70)
            
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

