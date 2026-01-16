#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Kalman Filter Implementation for ICM-20948 IMU
This demonstrates how to use a Kalman filter to fuse accelerometer and gyroscope data
for better orientation estimation.

The Kalman filter provides optimal estimation by:
1. Predicting state based on gyroscope (process model)
2. Correcting with accelerometer measurements (observation model)
3. Adaptively weighting predictions vs measurements based on uncertainty
"""

import time
import math
import numpy as np
import sys
import os

# Add the ICM20948 module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ICM-20948', 'Raspberry Pi', 'python'))
import ICM20948

# Access global variables from the module
Accel = ICM20948.Accel
Gyro = ICM20948.Gyro
Mag = ICM20948.Mag

class KalmanFilter:
    """
    Simple Kalman Filter for 2D orientation (pitch and roll)
    Can be extended to 3D (pitch, roll, yaw) if magnetometer is used
    """
    
    def __init__(self):
        # State vector: [pitch, roll] (in radians)
        self.state = np.array([[0.0], [0.0]])  # Initial pitch and roll
        
        # State covariance matrix (uncertainty in our estimate)
        # Larger values = more uncertainty
        self.P = np.eye(2) * 0.1
        
        # Process noise covariance (how much we trust our process model)
        # Represents uncertainty in gyroscope integration
        self.Q = np.eye(2) * 0.001
        
        # Measurement noise covariance (how much we trust accelerometer)
        # Larger values = less trust in measurements
        self.R = np.eye(2) * 0.1
        
        # Last update time (for computing dt)
        self.last_time = time.time()
        
    def predict(self, gyro_x, gyro_y, gyro_z, dt):
        """
        Predict step: Use gyroscope to predict new orientation
        
        Args:
            gyro_x, gyro_y, gyro_z: Gyroscope rates in degrees/s
            dt: Time step in seconds
        """
        # Convert gyro rates from deg/s to rad/s
        gyro_x_rad = math.radians(gyro_x)
        gyro_y_rad = math.radians(gyro_y)
        
        # State transition: new_pitch = old_pitch + pitch_rate * dt
        # For small angles: pitch_rate ≈ gyro_y, roll_rate ≈ gyro_x
        # (This is a simplified model - for large angles, need rotation matrices)
        F = np.eye(2)  # State transition matrix (identity for linear integration)
        
        # Update state: x_k = F * x_{k-1} + B * u_k
        # u_k is the gyro input: [gyro_y_rad, gyro_x_rad]
        B = np.array([[dt, 0], [0, dt]])  # Input matrix
        u = np.array([[gyro_y_rad], [gyro_x_rad]])
        
        self.state = F @ self.state + B @ u
        
        # Update covariance: P_k = F * P_{k-1} * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, accel_x, accel_y, accel_z):
        """
        Update step: Use accelerometer to correct orientation estimate
        
        Args:
            accel_x, accel_y, accel_z: Accelerometer readings in g's
        """
        # Compute pitch and roll from accelerometer
        # When stationary, accelerometer measures gravity
        accel_norm = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        if accel_norm < 0.1:  # If acceleration is too small, skip update
            return
        
        # Normalize
        accel_x /= accel_norm
        accel_y /= accel_norm
        accel_z /= accel_norm
        
        # Compute pitch and roll from accelerometer
        # Pitch: rotation around Y axis
        pitch_accel = math.asin(-accel_x)
        
        # Roll: rotation around X axis
        roll_accel = math.atan2(accel_y, accel_z)
        
        # Measurement vector
        z = np.array([[pitch_accel], [roll_accel]])
        
        # Measurement matrix (maps state to measurements)
        # In this case, we directly measure pitch and roll, so H = identity
        H = np.eye(2)
        
        # Innovation (measurement residual)
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(2) - K @ H) @ self.P
        
    def get_orientation(self):
        """
        Get current orientation estimate
        
        Returns:
            pitch, roll in degrees
        """
        pitch_rad = self.state[0, 0]
        roll_rad = self.state[1, 0]
        
        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)
        
        return pitch_deg, roll_deg
    
    def get_orientation_rad(self):
        """
        Get current orientation estimate in radians
        
        Returns:
            pitch, roll in radians
        """
        return self.state[0, 0], self.state[1, 0]


def read_acceleration(icm):
    """
    Read and convert acceleration measurements from ICM-20948
    
    Args:
        icm: ICM20948 sensor object
        
    Returns:
        accel_x, accel_y, accel_z in g's
    """
    # Read sensor data
    icm.icm20948_Gyro_Accel_Read()
    
    # Raw acceleration values (LSB counts)
    accel_x_raw = Accel[0]
    accel_y_raw = Accel[1]
    accel_z_raw = Accel[2]
    
    # Convert to g's
    # For ±2g range: 16384 LSB/g
    # For ±4g range: 8192 LSB/g
    # For ±8g range: 4096 LSB/g
    # For ±16g range: 2048 LSB/g
    ACCEL_SCALE = 16384.0  # Current configuration is ±2g
    
    accel_x_g = accel_x_raw / ACCEL_SCALE
    accel_y_g = accel_y_raw / ACCEL_SCALE
    accel_z_g = accel_z_raw / ACCEL_SCALE
    
    return accel_x_g, accel_y_g, accel_z_g


def read_gyroscope(icm):
    """
    Read and convert gyroscope measurements from ICM-20948
    
    Args:
        icm: ICM20948 sensor object
        
    Returns:
        gyro_x, gyro_y, gyro_z in degrees/s
    """
    # Read sensor data (this also reads gyro)
    icm.icm20948_Gyro_Accel_Read()
    
    # Raw gyroscope values (LSB counts)
    # Gyro values are already offset-corrected in icm20948_Gyro_Accel_Read()
    gyro_x_raw = Gyro[0]
    gyro_y_raw = Gyro[1]
    gyro_z_raw = Gyro[2]
    
    # Convert to degrees/s
    # For ±1000 DPS range: 32.8 LSB/(deg/s)
    # For ±500 DPS range: 65.5 LSB/(deg/s)
    # For ±250 DPS range: 131 LSB/(deg/s)
    # For ±2000 DPS range: 16.4 LSB/(deg/s)
    GYRO_SCALE = 32.8  # Current configuration is ±1000 DPS
    
    gyro_x_dps = gyro_x_raw / GYRO_SCALE
    gyro_y_dps = gyro_y_raw / GYRO_SCALE
    gyro_z_dps = gyro_z_raw / GYRO_SCALE
    
    return gyro_x_dps, gyro_y_dps, gyro_z_dps


if __name__ == '__main__':
    print("\n=== Kalman Filter IMU Test Program ===\n")
    
    # Initialize sensor
    try:
        icm = ICM20948.ICM20948()
        print("ICM-20948 initialized successfully\n")
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        exit(1)
    
    # Initialize Kalman filter
    kf = KalmanFilter()
    
    print("Starting Kalman filter loop...")
    print("Press Ctrl+C to stop\n")
    print("=" * 70)
    
    try:
        while True:
            # Get current time for dt calculation
            current_time = time.time()
            dt = current_time - kf.last_time
            kf.last_time = current_time
            
            # Read sensors
            accel_x, accel_y, accel_z = read_acceleration(icm)
            gyro_x, gyro_y, gyro_z = read_gyroscope(icm)
            
            # Kalman filter steps
            # 1. Predict using gyroscope
            kf.predict(gyro_x, gyro_y, gyro_z, dt)
            
            # 2. Update using accelerometer
            kf.update(accel_x, accel_y, accel_z)
            
            # Get filtered orientation
            pitch_kf, roll_kf = kf.get_orientation()
            
            # For comparison: compute pitch/roll directly from accelerometer
            accel_norm = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            if accel_norm > 0.1:
                pitch_accel = math.degrees(math.asin(-accel_x / accel_norm))
                roll_accel = math.degrees(math.atan2(accel_y / accel_norm, accel_z / accel_norm))
            else:
                pitch_accel = 0
                roll_accel = 0
            
            # Display results
            print(f"\nTime: {current_time:.2f}s, dt: {dt:.3f}s")
            print(f"Acceleration (g): X={accel_x:7.3f}, Y={accel_y:7.3f}, Z={accel_z:7.3f}")
            print(f"Gyroscope (deg/s): X={gyro_x:7.2f}, Y={gyro_y:7.2f}, Z={gyro_z:7.2f}")
            print(f"Kalman Filter: Pitch={pitch_kf:7.2f}°, Roll={roll_kf:7.2f}°")
            print(f"Accelerometer: Pitch={pitch_accel:7.2f}°, Roll={roll_accel:7.2f}°")
            print("-" * 70)
            
            time.sleep(0.1)  # 10 Hz update rate
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

