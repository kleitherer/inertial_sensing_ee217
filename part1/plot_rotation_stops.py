import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CHOOSE AXIS: Set to "x" or "y"
# ============================================================
AXIS = "x"  # Change this to "y" to analyze Y-axis rotation

# Configuration based on axis selection
if AXIS == "x":
    CSV_PATH = "rotation_around_x_0_to_180.csv"
    T_PICK_S = np.array([3.04, 14.55, 23.69, 33.2, 46.59], dtype=float)
elif AXIS == "y":
    CSV_PATH = "rotation_around_y_0_to_180.csv"
    T_PICK_S = np.array([3.03, 14.41, 23.86, 37.57, 50.05], dtype=float)
else:
    raise ValueError("AXIS must be 'x' or 'y'")

# Ground-truth angles (same for both axes)
GT_DEG = np.array([0, 45, 90, 135, 180], dtype=float)


def unwrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    """Unwrap a degree-valued angle trace so it does not jump at +/-180."""
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def summarize_metrics(sample_table: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Compute accuracy metrics (in degrees) for gyro and accel."""
    out_rows = []

    for name in ["gyro", "accel"]:
        # signed error is useful to see bias; absolute error is for MAE/max
        e = (sample_table[f"{name}_angle_deg"] - sample_table["ground_truth_deg"]).to_numpy(dtype=float)
        ae = np.abs(e)

        out_rows.append({
            "set": tag,
            "sensor": name,
            "N": int(np.sum(np.isfinite(e))),
            "mean_signed_error_deg": float(np.nanmean(e)),
            "MAE_deg": float(np.nanmean(ae)),
            "RMSE_deg": float(np.sqrt(np.nanmean(e**2))),
            "max_abs_error_deg": float(np.nanmax(ae)),
        })

    return pd.DataFrame(out_rows)


def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # elapsed time from start (s)
    t_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()

    # angles from the CSV (deg)
    ax_deg = df[f"accel_angle_around_{AXIS}"].to_numpy(dtype=float)
    gx_deg = df[f"gyro_angle_around_{AXIS}"].to_numpy(dtype=float)

    # unwrapped versions (better for plotting around +/-180)
    ax_u = unwrap_degrees(ax_deg)
    gx_u = unwrap_degrees(gx_deg)

    # ---------------------- plot ----------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_s, ax_u, label=f"Accel angle around {AXIS.upper()}", linewidth=1.6)
    ax.plot(t_s, gx_u, label=f"Gyro angle around {AXIS.upper()}", linewidth=1.2, alpha=0.9)

    for k, gt in enumerate(GT_DEG):
        ax.axhline(
            gt,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Ground truth (0/45/90/135/180°)" if k == 0 else None,
        )

    ax.set_title(f"Rotation around {AXIS.upper()}: 0 to 180 degrees")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()  # close the plot window to continue

    # ------------------ sample at chosen times ------------------
    if len(T_PICK_S) != len(GT_DEG):
        raise ValueError("T_PICK_S and GT_DEG must have the same length")

    rows = []
    for t_pick, gt in zip(T_PICK_S, GT_DEG):
        # choose the nearest saved sample in the CSV
        i = int(np.argmin(np.abs(t_s - t_pick)))

        rows.append({
            "ground_truth_deg": float(gt),
            "t": float(t_pick),
            "gyro_angle_deg": float(gx_u[i]),
            "accel_angle_deg": float(ax_u[i]),
        })

    pts = pd.DataFrame(rows)

    # per-point errors
    for name in ["gyro", "accel"]:
        signed_err = pts[f"{name}_angle_deg"] - pts["ground_truth_deg"]
        pts[f"{name}_err_deg"] = signed_err
        pts[f"{name}_abs_err_deg"] = np.abs(signed_err)

        # percent error is not defined at gt=0, so keep NaN there
        pts[f"{name}_pct_err"] = np.where(
            pts["ground_truth_deg"] != 0,
            pts[f"{name}_abs_err_deg"] / pts["ground_truth_deg"] * 100.0,
            np.nan,
        )

    print("\nPoint-sampled angles + errors:")
    print(
        pts[
            [
                "ground_truth_deg",
                "t",
                "gyro_angle_deg",
                "accel_angle_deg",
                "gyro_err_deg",
                "accel_err_deg",
                "gyro_abs_err_deg",
                "accel_abs_err_deg",
                "gyro_pct_err",
                "accel_pct_err",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:8.3f}")
    )

    pts.to_csv(f"gt_point_errors_{AXIS}.csv", index=False)

    # ------------------ summary metrics ------------------
    # NOTE: these metrics are computed in DEGREES (not percent)
    metrics = summarize_metrics(pts, tag="all")

    metrics.to_csv(f"gt_point_metrics_{AXIS}.csv", index=False)

    print("\nSummary metrics (degrees):")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))



if __name__ == "__main__":
    main()
