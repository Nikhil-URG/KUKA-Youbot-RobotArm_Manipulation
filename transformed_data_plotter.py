# transformed_data_plotter.py
# Modified version of data_plotter.py that applies coordinate transformation to OptiTrack data
# to align frames and replots in a separate folder.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re
import os

ROOT = Path(__file__).parent.resolve()
print(f"Working from: {ROOT}")

# ==================== LOAD OPTITRACK ====================
def load_optitrack():
    data = []
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{1,2}\.\d{2}\.\d{2} [AP]M)")
    for size_key, size_folder in {"small": "Small", "medium": "Medium", "large": "Large"}.items():
        for dir_key, dir_folder in {"left": "Left", "right": "Right", "straight": "Straight"}.items():
            folder = ROOT / size_folder / dir_folder
            if not folder.exists(): continue
            for file in folder.glob("Take *.csv"):
                m = pattern.search(file.name)
                if not m: continue
                dt = datetime.strptime(m.group(1), "%Y-%m-%d %I.%M.%S %p")

                df = pd.read_csv(file, skiprows=7)
                if len(df) < 5: continue
                last = df.iloc[-1]

                # Quaternion to yaw
                try:
                    qx = float(last["X"])
                    qy = float(last["Y"])
                    qz = float(last["Z"])
                    qw = float(last["W"])
                    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
                except:
                    yaw = 0.0

                data.append({
                    "size": size_key, "direction": dir_key, "datetime": dt,
                    "x": float(last["X.1"]), "y": float(last["Y.1"]), "theta": yaw
                })
    return pd.DataFrame(data)

# ==================== LOAD YOUBOT ====================
def load_youbot():
    data = []
    group3 = ROOT / "Group3"
    for size_dir in group3.iterdir():
        if not size_dir.is_dir() or "_" not in size_dir.name: continue
        size, direction = size_dir.name.split("_", 1)
        csv_folder = size_dir / "csv"
        if not csv_folder.exists(): continue

        for csv_file in csv_folder.glob("2025_11_27_*.csv"):
            stem = csv_file.stem
            time_str = "_".join(stem.split("_")[:6])
            try:
                dt = datetime.strptime(time_str, "%Y_%m_%d_%H_%M_%S")
            except:
                continue

            df = pd.read_csv(csv_file, header=None, names=["x","y","theta"])
            if df.empty: continue
            last = df.iloc[-1]

            data.append({
                "size": size, "direction": direction, "datetime": dt,
                "x": float(last.x) * 100, "y": float(last.y) * 100, "theta": float(last.theta)
            })
    return pd.DataFrame(data)

# ==================== OUTLIER REMOVAL (IQR) ====================
def remove_outliers_per_group(df):
    cleaned_groups = []
    outlier_counts = {}

    for (size, direction), group in df.groupby(["size", "direction"]):
        key = f"{size}_{direction}"
        if len(group) < 4:
            cleaned_groups.append(group)
            outlier_counts[key] = 0
            continue

        Q1 = group[["x_youbot", "y_youbot"]].quantile(0.25)
        Q3 = group[["x_youbot", "y_youbot"]].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((group[["x_youbot", "y_youbot"]] < (Q1 - 1.5*IQR)) |
                 (group[["x_youbot", "y_youbot"]] > (Q3 + 1.5*IQR))).any(axis=1)

        outliers = group[~mask]
        outlier_counts[key] = len(outliers)
        cleaned_groups.append(group[mask])

    # Print summary
    print("\nOutlier removal summary:")
    total = 0
    for key in sorted(outlier_counts):
        cnt = outlier_counts[key]
        s, d = key.split("_")
        print(f"   {s.capitalize():6} {d.capitalize():9} : {cnt:2d} outliers removed")
        total += cnt
    print(f"   TOTAL outliers removed : {total}\n")

    return pd.concat(cleaned_groups, ignore_index=True), outlier_counts

# ==================== MAIN ====================
print("Loading data...")
opti_df   = load_optitrack()
youbot_df = load_youbot()
print(f"OptiTrack trials : {len(opti_df)}")
print(f"YouBot trials    : {len(youbot_df)}")

# Merge with proper suffixes
merged = pd.merge_asof(
    youbot_df.sort_values("datetime"),
    opti_df.sort_values("datetime"),
    by=["size", "direction"],
    on="datetime",
    tolerance=pd.Timedelta(minutes=10),
    direction="nearest",
    suffixes=("_youbot", "_opti")
)

matched = merged.dropna(subset=["x_opti", "y_opti"]).copy()
print(f"Successfully matched {len(matched)} trials\n")

# Apply coordinate transformation to YouBot data
# World origin [0, 0, 0] in robot frame corresponds to [-212.40, 76.17, 103.15] in OptiTrack frame
# To transform YouBot (robot frame) to OptiTrack frame: add the offset
offset_x = -212.40
offset_y = 76.17
offset_z = 103.15  # Not used in 2D plotting, but for completeness
matched['x_youbot'] = matched['x_youbot'] + offset_x
matched['y_youbot'] = matched['y_youbot'] + offset_y
print("Applied coordinate transformation to YouBot data.")

# Create output folder
output_folder = ROOT / "transformed_plots"
os.makedirs(output_folder, exist_ok=True)

# ==================== CREATE 3 COMBINED PLOTS ====================

colors = {'straight': 'blue', 'left': 'green', 'right': 'red'}
markers = {'straight': 'o', 'left': '^', 'right': 's'}

for size in sorted(matched['size'].unique()):
    plt.figure(figsize=(12, 10))

    for direction in ['straight', 'left', 'right']:
        group = matched[(matched['size'] == size) & (matched['direction'] == direction)]

        # YouBot (internal, transformed) - single point per direction
        if not group.empty:
            youbot_x = group.x_youbot.iloc[0]
            youbot_y = group.y_youbot.iloc[0]
            youbot_theta = group.theta_youbot.iloc[0]
            plt.scatter(youbot_x, youbot_y, c=colors[direction], s=200, marker=markers[direction],
                       label=f"YouBot {direction} (Transformed)", edgecolors='black', linewidth=1.5, zorder=10)
            # Arrow for orientation
            dx = 8 * np.sin(youbot_theta)
            dy = 8 * np.cos(youbot_theta)
            plt.arrow(youbot_x, youbot_y, dx, dy, head_width=3, color=colors[direction],
                     length_includes_head=True, zorder=15)

        # OptiTrack (external)
        plt.scatter(group.x_opti, group.y_opti, c=colors[direction], s=120, marker='X',
                   label=f"OptiTrack {direction}", zorder=5)
        # Arrows for OptiTrack
        for _, r in group.iterrows():
            dx = 6 * np.sin(r.theta_opti)
            dy = 6 * np.cos(r.theta_opti)
            plt.arrow(r.x_opti, r.y_opti, dx, dy, head_width=2, color=colors[direction],
                     alpha=0.7, length_includes_head=True, zorder=20)

    plt.xlabel("X [cm]", fontsize=14)
    plt.ylabel("Y – Forward direction [cm]", fontsize=14)
    plt.title(f"Final End-Effector and Object Poses\n(YouBot Transformed to OptiTrack Frame)\n{size.capitalize()} Object – All Directions\n"
             , fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=12, loc='upper left')

    filename = output_folder / f"{size}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved transformed plot: {filename.name}")

# Compute accuracy and precision with YouBot transformed to OptiTrack frame
print("\nComputing statistical analysis with YouBot transformed to OptiTrack frame...")

accuracy = {}
precision = {}

for (size, direction), group in matched.groupby(["size", "direction"]):
    print(f"Group {size}_{direction}: {len(group)} trials")
    print(f"  x_youbot range: {group.x_youbot.min():.2f} to {group.x_youbot.max():.2f}")
    print(f"  y_youbot range: {group.y_youbot.min():.2f} to {group.y_youbot.max():.2f}")
    # Accuracy: mean distance between youbot and transformed opti
    distances = np.sqrt((group.x_youbot - group.x_opti)**2 + (group.y_youbot - group.y_opti)**2)
    accuracy[f"{size}_{direction}"] = distances.mean()

    # Precision: std dev of transformed opti positions (repeatability)
    std_x = group["x_opti"].std()
    std_y = group["y_opti"].std()
    precision[f"{size}_{direction}"] = (std_x + std_y) / 2

print("\nAccuracy (mean distance to ground truth, cm):")
for key, val in accuracy.items():
    print(f"  {key}: {val:.2f}")

print("\nPrecision (mean std dev of positions, cm):")
for key, val in precision.items():
    print(f"  {key}: {val:.2f}")

# Save transformed preprocessed data as CSV
preprocessed = []
trial_counter = {}
for (size, direction), group in matched.groupby(["size", "direction"]):
    if f"{size}_{direction}" not in trial_counter:
        trial_counter[f"{size}_{direction}"] = 1
    for idx, row in group.iterrows():
        trial = trial_counter[f"{size}_{direction}"]
        trial_counter[f"{size}_{direction}"] += 1
        # External (transformed)
        preprocessed.append({
            "mass": size,
            "position": direction,
            "trial": trial,
            "type": "external",
            "x": row.x_opti,
            "y": row.y_opti,
            "theta": row.theta_opti
        })
        # Internal
        preprocessed.append({
            "mass": size,
            "position": direction,
            "trial": trial,
            "type": "internal",
            "x": row.x_youbot,
            "y": row.y_youbot,
            "theta": row.theta_youbot
        })

preprocessed_df = pd.DataFrame(preprocessed)
transformed_csv_path = ROOT / "transformed_preprocessed_data.csv"
preprocessed_df.to_csv(transformed_csv_path, index=False)
print(f"\nSaved transformed preprocessed data to {transformed_csv_path}")

print("\nAll transformed plots saved successfully!")