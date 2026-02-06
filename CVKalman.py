#!/Users/s/bin/python3
# Simple constant-velocity Kalman filter, Sam, 10/01/2023
# Airplane is flying at constant velocity, and our only measurement is 
# a noisy radar signal of position
# Note that all numbers in matrices have a decimal point to force float
# For simplicity, distances are in meters, time is in seconds (velocity is m/s),
# and our sampling rate is 1Hz (so 1 second between samples)
# 
# Invocation:
# CVKalman.py <noisevar>
# where <noisevar> is the noise variance of the sensor (in m^2)
# Default noisevar is 10^6
#
# Example: CVKalman.py 1000.0
#


import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

#
# Simulate sensor output
# Airplane is flying at 300m/s (670mph); simply compute true position and add noise to it
# Arguments:
#    t: time (in seconds)
#    var: noise variance in measurement
#    trueV: true velocity
#
def read_noisy_radar(t,var,trueV):
    noise = np.random.normal(0.0,m.sqrt(var))
    return(noise + trueV*t)
    
trueVelocity = 300.0                    # Ground truth velocity

measVar = 1.0e6                        # noise is ~1km
if len(sys.argv) > 1:
    measVar = float(sys.argv[1])



kf = KalmanFilter(dim_x=2,dim_z=1)      # state vector is [x v], v=xdot=velocity

#
# Construct state transition matrix and measurement array
# 
kf.F = np.array([[1.,1.],               # 1.0 in upper right corresponds to 1Hz (deltat = 1sec)
                 [0.,1.]])
kf.H = np.array([[1.,0.]])

#
# Initial conditions; take a guess that initial velocity is 200m/s, initial position is zero
# Wild-ass guess on initial covariance matrix (10000m^2 in position, 10000(m/s)^2 in velocity)
#
kf.x = np.array([0., 200.])
kf.P = np.array([[10000.,  0.],
                 [ 0., 10000.]])

kf.Q = np.array([[0.,  0.],
                 [0.,  0.]])            # No process noise (very stable airplane)

kf.R = np.array([[ measVar ]])          # Variance of measurement


# Utility arrays to save off Kalman state over time
xs=[];
vs=[];
zs=[];
kgs=[];

stopTime = 100
taxis = range(stopTime)
for t in taxis:
    z=read_noisy_radar(t,measVar,trueVelocity) 
    kf.predict()                        # State prediction
    kf.update(z)                        # Update
    kgs.append(kf.K)                    # Save off Kalman gain
    zs.append(z)                        # Save off sensor output
    xs.append(kf.x[0])                  # Save off position estimate
    vs.append(kf.x[1])                  # Save off velocity estimate



# To compare: exceedingly naive filtering
# Compute simple moving average filter on position
ma_len = 10                            # Length of moving average
delayz = ([0]*ma_len) + zs             # Pad with initial zeros to simplify
filt_zs = [ sum(delayz[i:i+ma_len])/ma_len for i in range(stopTime) ]   # Filter data using list comp


# Plot-o-rama
plt.figure()
plt.plot(taxis,vs,label='KF velocity estimate')
plt.plot(taxis,np.full(stopTime,trueVelocity),label='True velocity')
plt.grid()
plt.legend()

plt.figure()
groundTruth=[trueVelocity*i for i in range(stopTime)]
plt.plot(taxis,xs,label='KF position estimate') 
plt.plot(taxis,zs,label='Sensor output')       
plt.plot(taxis,groundTruth,label='Ground truth')
plt.grid()
plt.legend()

plt.figure()            
plt.plot(taxis,[x[0,0] for x in kgs],label='Kalman gain')  # Plot Kalman gain vs time
plt.grid()
plt.legend()

plt.figure()            
plt.plot(taxis,groundTruth,label='Ground truth')
plt.plot(taxis,xs,label='KF position estimate') 
plt.plot(taxis,filt_zs,label='Simple 10 pt moving average filter')
plt.grid()
plt.legend()

plt.show()
