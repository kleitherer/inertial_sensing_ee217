#!/usr/bin/env python3
# EE217 Lab 1 - Part 3(b)
# Constant-velocity KF with a known velocity "command" step at t=500.
# State: x = [position, velocity]^T
# Measurement: z = position (noisy radar)
#
# Control model:
#   x_{k+1} = F x_k + B u_k + w_k
#   z_k     = H x_k + v_k
# where u_k is a known input we apply (here: +100 m/s at t=500 only).

import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

STEP_T = 500
V1 = 300.0
V2 = 400.0

def true_velocity(t):
    return V1 if t <= STEP_T else V2

def true_position(t):
    if t <= STEP_T:
        return V1 * t
    return V1 * STEP_T + V2 * (t - STEP_T)

def read_noisy_radar(t, meas_var):
    noise = np.random.normal(0.0, m.sqrt(meas_var))
    return true_position(t) + noise

def reconverge_samples(v_est, t_list, thresh=25.0, streak_len=10, t_start=STEP_T):
    streak = 0
    for i in range(len(t_list)):
        t = t_list[i]
        if t < t_start:
            continue
        err = abs(v_est[i] - true_velocity(t))
        if err < thresh:
            streak += 1
            if streak >= streak_len:
                return t - t_start  # samples after step
        else:
            streak = 0
    return None

def run_case(measVar):
    kf = KalmanFilter(dim_x=2, dim_z=1)

    dt = 1.0 
    kf.F = np.array([[1., dt],
                     [0., 1.]])

    kf.H = np.array([[1., 0.]])

    # B is the "G matrix" from lecture (control input matrix).
    # Here u_k is a velocity increment (m/s). It directly changes velocity, not position.
    kf.B = np.array([[0.],
                     [1.]])

    kf.x = np.array([[0.],
                     [200.]])
    kf.P = np.array([[10000., 0.],
                     [0., 10000.]])

    kf.Q = np.zeros((2, 2))
    kf.R = np.array([[measVar]])

    taxis, vhat, vgt = [], [], []

    for t in range(5000):
        taxis.append(t)

        z = read_noisy_radar(t, measVar)

        # Control input: apply +100 m/s ONLY at the step time.
        if t == STEP_T:
            u = np.array([[V2 - V1]])   # +100
        else:
            u = np.array([[0.]])

        kf.predict(u=u)
        kf.update(np.array([[z]]))

        vhat.append(float(kf.x[1, 0]))
        vgt.append(true_velocity(t))

    n_samp = reconverge_samples(vhat, taxis)
    return np.array(taxis), np.array(vhat), np.array(vgt), n_samp

def main():
    noise = [1e3, 1e7, 1e9]

    plt.figure(figsize=(10, 5))
    results = []

    for R in noise:
        t, vhat, vgt, n_samp = run_case(R)
        results.append((R, n_samp))
        plt.plot(t, vhat, label=f"v_hat, R={R:.0e}")

    plt.plot(t, vgt, "k", linewidth=2, label="True velocity")
    plt.axvline(STEP_T, linestyle="--", label=f"Step at t={STEP_T}")
    plt.grid(True)
    plt.legend()
    plt.title("Part 3(b): Velocity tracking")
    plt.xlabel("t [samples] (dt=1s)")
    plt.ylabel("velocity [m/s]")
    plt.tight_layout()
    plt.show()

    print("Congerences for < 25 for 10 consecutive samples):")
    for R, n_samp in results:
        print(f"  R={R:.0e}: {n_samp}")

if __name__ == "__main__":
    main()