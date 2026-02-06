"""
Consider the problem of predicting the remaining charge (in coulombs) in a smartphone battery.
We can measure the current draw from the battery using a current meter (measuring
coulombs/second, or amps), but the noise in the current meter is quite high – its noise deviation
is 100mA. For simplicity, assume the sensor samples once every second.

If we know that the full charge of the battery is 5000mA-hr, and that the drain on the
battery is constant but unknown, build a Kalman filter that estimate the remaining charge
on the battery and the drain rate given the noisy current meter. The output of the
Kalman should be the percentage of battery charge remaining. By “build a KF,” specify
the matrices that define the dynamics of the KF that would be appropriate for, say, use in
the filterpy.KalmanFilter code (x, H, R, P, F, Q). (10 points
"""
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

np.random.seed(0)

dt = 1.0
T = 1000

true_current = 0.200          # A
meas_std = 0.100              # A
meas_var = meas_std**2

full_charge = 5.0 * 3600.0    # 5000 mA-hr in Coulombs

def measure_current():
    return true_current + np.random.normal(0.0, meas_std)


"""
These are defined in part a
"""
kf = KalmanFilter(dim_x=2, dim_z=1)

kf.F = np.array([
    [1.0, -dt],
    [0.0,  1.0]
])

kf.H = np.array([[0.0, 1.0]])

kf.R = np.array([[meas_var]])
kf.Q = np.zeros((2,2))

kf.x = np.array([
    [full_charge],
    [0.5]  
])

kf.P = np.array([
    [(0.1 * full_charge)**2, 0.0],
    [0.0, (0.5)**2]
])


"""
The following values are to help plot part b and determine runtime.
"""
t = np.arange(T + 1)

q_true = np.zeros_like(t, dtype=float)
q_est  = np.zeros_like(t, dtype=float)

pct_true = np.zeros_like(t, dtype=float)
pct_est  = np.zeros_like(t, dtype=float)

i_est = np.zeros_like(t, dtype=float)

q_true[0] = full_charge


for k in range(len(t)):
    if k > 0:
        q_true[k] = max(0.0, q_true[k-1] - true_current * dt)
    z = measure_current()
    kf.predict()
    kf.update([[z]])

    qk = float(kf.x[0])
    ik = float(kf.x[1])

    qk = min(max(qk, 0.0), full_charge)

    q_est[k] = qk
    i_est[k] = ik

    pct_true[k] = 100.0 * q_true[k] / full_charge
    pct_est[k]  = 100.0 * qk / full_charge

runtime_true = full_charge / true_current
runtime_kf   = full_charge / max(i_est[-1], 1e-6)

plt.figure(figsize=(10,4))
plt.plot(t, pct_true, label="True battery %")
plt.plot(t, pct_est, label="KF estimated battery %")
plt.xlabel("Time [s]")
plt.ylabel("Battery [%]")
plt.title("Battery charge estimation with Kalman Filter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"True drain current: {true_current*1000:.1f} mA")
print(f"KF estimated current at t=1000 s: {i_est[-1]*1000:.2f} mA\n")

print("Estimated runtime (full → empty):")
print(f"  True current: {runtime_true:.1f} s = {runtime_true/3600:.3f} hr")
print(f"  KF estimate : {runtime_kf:.1f} s = {runtime_kf/3600:.3f} hr")