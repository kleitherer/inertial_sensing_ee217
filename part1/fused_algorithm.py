import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Pick axis
# ============================================================
AXIS = "y"  # select rotation around "x" or "y" axis
TAU_S = 2 # increasing tau_s increases the gyroscope weight

if AXIS == "x":
    CSV_PATH = "rotation_around_x_0_to_180.csv"
    TITLE_AXIS = "X"
    T_PICK_S = np.array([3.04, 14.55, 23.69, 33.2, 46.59], dtype=float)
elif AXIS == "y":
    CSV_PATH = "rotation_around_y_0_to_180.csv"
    TITLE_AXIS = "Y"
    T_PICK_S = np.array([3.03, 14.41, 23.86, 37.57, 50.05], dtype=float)
else:
    raise ValueError("AXIS must be 'x' or 'y'")

GT_DEG = np.array([0, 45, 90, 135, 180], dtype=float)

def wrap_deg(a):
    """Wrap to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0

def angdiff_deg(a, b):
    """Smallest signed difference a-b in degrees, in [-180,180)."""
    return wrap_deg(a - b)

def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Elapsed time from start (for plotting/sampling, not used in fusion math)
    t_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy(float)

    # Get metrics from CSV
    dt = df["dt"].to_numpy(float) 
    alpha = TAU_S / (TAU_S + dt)

    gyro = df[f"gyro_angle_around_{AXIS}"].to_numpy(float)
    accel = df[f"accel_angle_around_{AXIS}"].to_numpy(float)

    # Complementary filter
    fused = np.empty_like(gyro)
    fused[0] = gyro[0]

    for k in range(1, len(fused)):
        # gyro increment
        dgyro = angdiff_deg(gyro[k], gyro[k - 1])

        # predict
        pred = wrap_deg(fused[k - 1] + dgyro)

        # accel correction uses wrap-safe error
        err = angdiff_deg(accel[k], pred)  # accel - pred on circle

        # update: pred + (1-alpha)*err  (equivalent to alpha*pred + (1-alpha)*accel on circle)
        fused[k] = wrap_deg(pred + (1.0 - alpha[k]) * err)

    print("AXIS:", AXIS, "CSV:", CSV_PATH)
    print("TAU_S:", TAU_S)
    print("dt median/min/max:", float(np.median(dt)), float(np.min(dt)), float(np.max(dt)))
    print("alpha min/max:", float(np.min(alpha)), float(np.max(alpha)))

    # For plots, unwrap to make steps continuous-looking
    gyro_u = np.rad2deg(np.unwrap(np.deg2rad(gyro)))
    accel_u = np.rad2deg(np.unwrap(np.deg2rad(accel)))
    fused_u = np.rad2deg(np.unwrap(np.deg2rad(fused)))

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, accel_u, label=f"accel_angle_around_{AXIS} (CSV)", linewidth=1.8)
    plt.plot(t_s, gyro_u, label=f"gyro_angle_around_{AXIS} (CSV)", linewidth=1.3, alpha=0.85)
    plt.plot(t_s, fused_u, label=f"fused (circular CF, tau={TAU_S:.3g}s)", linewidth=2.2)
    
    # Add ground truth lines for the angles
    for k, gt in enumerate(GT_DEG):
        plt.axhline(
            gt,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Ground truth (0/45/90/135/180°)" if k == 0 else None,
        )
    
    plt.title(f"Sensor fusion results for rotation around {TITLE_AXIS} axis")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3.2))
    plt.plot(t_s, alpha, linewidth=1.5)
    plt.title("Alpha over time (gyro weight)")
    plt.xlabel("Time (s)")
    plt.ylabel("alpha")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Point sampling + wrapped errors
    rows = []
    for t_pick, gt in zip(T_PICK_S, GT_DEG):
        i = int(np.argmin(np.abs(t_s - t_pick)))
        rows.append({
            "ground_truth_deg": float(gt),
            "t": float(t_pick),
            "alpha_used": float(alpha[i]),
            "gyro_angle_deg": float(gyro_u[i]),
            "accel_angle_deg": float(accel_u[i]),
            "fused_angle_deg": float(fused_u[i]),
        })
    pts = pd.DataFrame(rows)

    for name in ["gyro", "accel", "fused"]:
        err = pts[f"{name}_angle_deg"].to_numpy(float) - pts["ground_truth_deg"].to_numpy(float)
        pts[f"{name}_err_deg"] = wrap_deg(err)
        pts[f"{name}_abs_err_deg"] = np.abs(pts[f"{name}_err_deg"])

    print("\nPoint-sampled angles + wrapped errors:")
    print(pts.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

if __name__ == "__main__":
    main()
