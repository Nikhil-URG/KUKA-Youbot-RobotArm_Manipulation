# separate_plots_generator.py
# Generates separate scatter plots for OptiTrack data, YouBot data, and combined transformed data

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
offset_x = -212.40
offset_y = 76.17
matched['x_youbot'] = matched['x_youbot'] + offset_x
matched['y_youbot'] = matched['y_youbot'] + offset_y
print("Applied coordinate transformation to YouBot data.")

# Create output folder
output_folder = ROOT / "report_plots"
os.makedirs(output_folder, exist_ok=True)

colors = {'straight': 'blue', 'left': 'green', 'right': 'red'}
markers = {'straight': 'o', 'left': '^', 'right': 's'}

# ==================== OPTITRACK SCATTER PLOTS ====================
for size in sorted(matched['size'].unique()):
    plt.figure(figsize=(10, 8))
    for direction in ['straight', 'left', 'right']:
        group = matched[(matched['size'] == size) & (matched['direction'] == direction)]
        plt.scatter(group.x_opti, group.y_opti, c=colors[direction], marker=markers[direction],
                   label=f"OptiTrack {direction}", s=50, alpha=0.7)
    plt.xlabel("X [cm]", fontsize=14)
    plt.ylabel("Y – Forward direction [cm]", fontsize=14)
    plt.title(f"OptiTrack Data Scatter Plot\n{size.capitalize()} Object – All Directions\n"
             , fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=12, loc='upper left')
    filename = output_folder / f"optitrack_{size}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved OptiTrack plot: {filename.name}")

# ==================== YOUBOT SCATTER PLOTS ====================
for size in sorted(matched['size'].unique()):
    plt.figure(figsize=(10, 8))
    for direction in ['straight', 'left', 'right']:
        group = matched[(matched['size'] == size) & (matched['direction'] == direction)]
        if not group.empty:
            youbot_x = group.x_youbot.iloc[0]
            youbot_y = group.y_youbot.iloc[0]
            plt.scatter(youbot_x, youbot_y, c=colors[direction], marker=markers[direction],
                       label=f"YouBot {direction}", s=200, edgecolors='black', linewidth=1.5)
    plt.xlabel("X [cm]", fontsize=14)
    plt.ylabel("Y – Forward direction [cm]", fontsize=14)
    plt.title(f"YouBot Data Scatter Plot\n{size.capitalize()} Object – All Directions\n"
             , fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=12, loc='upper left')
    filename = output_folder / f"youbot_{size}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved YouBot plot: {filename.name}")

# ==================== COMBINED TRANSFORMED SCATTER PLOTS ====================
for size in sorted(matched['size'].unique()):
    plt.figure(figsize=(12, 10))
    for direction in ['straight', 'left', 'right']:
        group = matched[(matched['size'] == size) & (matched['direction'] == direction)]
        # YouBot (transformed)
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
        # OptiTrack
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
    plt.title(f"Combined Transformed Data Scatter Plot\n(YouBot Transformed to OptiTrack Frame)\n{size.capitalize()} Object – All Directions\n"
             , fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=12, loc='upper left')
    filename = output_folder / f"combined_{size}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved combined plot: {filename.name}")

print("\nAll separate plots saved successfully!")