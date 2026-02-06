import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

def true_position(t):
    if t <= 500:
        return 300.0 * t
    else:
        return 300.0 * 500.0 + 400.0 * (t - 500)

def true_velocity(t):
    return 300.0 if t <= 500 else 400.0

def read_noisy_radar(t, var):
    noise = np.random.normal(0.0, m.sqrt(var))
    return true_position(t) + noise

def time_to_reconverge(v_est, t_list):
    streak = 0

    for i in range(len(t_list)):
        t = t_list[i]
        if t < 500:
            continue

        err = abs(v_est[i] - true_velocity(t))

        if err < 25.0:
            streak += 1
            if streak == 10:
                return t
        else:
            streak = 0

    return None


def run_case(measVar, stopTime):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.x = np.array([0., 200.])
    kf.P = np.array([[10000., 0.],
                     [0., 10000.]])
    kf.Q = np.array([[0., 0.],
                     [0., 0.]])
    kf.R = np.array([[measVar]])

    taxis = []
    vs = []
    vtrue = []

    for t in range(stopTime):
        taxis.append(t)

        z = read_noisy_radar(t, measVar) 

        kf.predict()
        kf.update(z)

        vs.append(kf.x[1])   
        vtrue.append(true_velocity(t)) 

    t_conv = time_to_reconverge(vs, taxis)
    return taxis, np.array(vs), np.array(vtrue), t_conv

noise = [1000, 10000000, 1000000000]
results = []

plt.figure()
for R in noise:
    taxis, vs, vtrue, t_conv = run_case(R, 5000)
    results.append((R, t_conv))
    plt.plot(taxis, vs, label=f"KF v-hat, R={R:.0e}")

plt.plot(taxis, vtrue, "k", linewidth=2, label="True velocity")
plt.axvline(500, linestyle="--")
plt.grid()
plt.legend()
plt.title("Velocity tracking after step change at t=500s")
plt.xlabel("t [s]")
plt.ylabel("velocity [m/s]")
plt.show()

print("Time until error < 25 for 10 samples:")
for R, t_conv in results:
    print(f"R={R:.0e}: t_conv={t_conv}")