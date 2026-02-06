"""
Using the accelerometer only, implement a 1-D Kalman filter to determine position based
on acceleration measurements in a single axis (say, the X axis only). This will be similar
in nature to the simple constant velocity example discussed in lecture, except that the
measurement is in acceleration and not position


You will need to take out the bias in he accelerometer (you can use the other axis of the accelerometer / gyro to do this);
you can either measure the bias and directly subtract it from the accelerometer output,
or implement a calibration step in your code before you start the position tracking.

Note that you will have to take out whatever gravity component is leaking into the
accelerometer measurement (and this is related to what you did in Part I)


Plot sensor error as a function of distance traveled; your ultimate performance metric is error
when the sensor is back at its original position (having traveled 6+6=12 feet). Submit
code and sensor error plot. Analyze your error sources; document what you did in your
code minimize to the error. Does how quickly you move impact your error?


this is how our data is stored: t_s,accel_x,accel_y,accel_z,gyro_x_raw,gyro_y_raw,gyro_z_raw,gyro_x_corr,gyro_y_corr,gyro_z_corr
"""


import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

"""
Define parameters of kalman filter 
"""

# state vector has the variables we want to track
# we are taking acceleration measurements, and we want to keep track of position and velocity

def find_accel():
    # go into csv and extract the right acceleration we want to use as our measurement 
    # remove gravity from the accelerometer to get linear acceleration along your axis
    # estimate orientation (roll/pitch) from gyro integrated and accel corrected
    # compute gravity vector in the sensor frame
    # subtract gravity component
    # subtract bias (either pre-calibrated or KF bias state
    return 0.0

dt = 0.1
T = len(csv) # replace with csv length

kf = KalmanFilter(dim_x=3, dim_z=1) #state vector is [x v a]

#
# Construct state transition matrix and measurement array
# 

# old one which doesn't use acceleration as measurement 
# kf.F = np.array([
#     [1,dt],
#     [0,1]
# ])

kf.F = np.array([
    [1,dt,0.5*(dt**2)],
    [0, 1, dt],
    [0, 0, 1]
])

# this is our measurement matrix, we're only measuring acceleration
kf.H = np.array([[0,0,1]])


# calculate variance of the measured acceleration... which we can get from where? 
# we have allen deviation plots, we have jsut lots of data
true_acceleration = 0
meas_std = 0.1
meas_var = meas_std**2
kf.R = np.array([[meas_var]])


# process noise covariance 
kf.Q = np.zeros((3,3))


# current state estimate... we want to start at x = 0, v = 0, a = 0?
# this is just a scalar for each position/velocity/acceleration, it doesn't save temporal data
kf.x = np.array([
    [0],
    [0],
    [0]
])

# state covariance matrix (error in estimate, (uncertainty in our system))
# initialize with this: we're most uncertain about position, then velocity, and least about acceleration
# TODO: adjust these values (theyre from chatGPT)
kf.P = np.diag([10.0**2, 1.0**2, 0.5**2])



"""
The following values are to help plot part b and determine runtime.
"""
t = np.arange(T + 1)

# we want to save the estimates at each timestep 

for k in range(len(t)):
    z = float(find_accel(k))
    kf.predict()
    kf.update(np.array([[z]]))

    # recompute each iteration
    x = float(kf.x[0])
    v = float(kf.x[1])
    a = float(kf.x[2])



plt.figure(figsize=(10,4))
# could create ground truth based on exact steps on blocks
# plt.plot(t, pct_true, label="True battery %")
plt.plot(t, x, label="KF estimated distance")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("1D position estimation with Kalman Filter using accelerometer")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# should we present as displacement or distance?... probably displacement

print(f"  True displacement: 0m")
print(f"Final position estimate: {x_hist[-1]:.3f} m")
print(f"Final velocity estimate: {v_hist[-1]:.3f} m/s")