# combined_data_plotter.py
# Combines data from all groups (1,2,3,4) and performs the same analysis and plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).parent.resolve()
print(f"Working from: {ROOT}")

# ==================== LOAD OPTITRACK COMBINED ====================
def load_optitrack_combined():
    data = []
    groups = {
        'Group1': ROOT / 'Group1' / 'optitrack',
        'Group2': ROOT / 'Group2' / 'motive',
        'Group3': ROOT,
        'Group4': ROOT / 'Group4' / 'SEE_Group4_2025_CSV_Opti_Track' / 'SEE_Group4_2025 - CSV'
    }

    for group_name, base_path in groups.items():
        if not base_path.exists():
            print(f"Skipping {group_name}: path not found")
            continue

        for size_key in ["small", "medium", "large"]:
            for dir_key in ["left", "right", "straight"]:
                if group_name == 'Group2':
                    folder = base_path / f"{size_key}_{dir_key}"
                elif group_name == 'Group1':
                    folder = base_path / size_key / dir_key
                elif group_name in ['Group3', 'Group4']:
                    folder = base_path / size_key.capitalize() / dir_key.capitalize()
                else:
                    continue

                if not folder.exists():
                    continue

                # Different file patterns
                if group_name == 'Group2':
                    pattern = re.compile(r"G2 \w\w Take (\d{4}-\d{2}-\d{2}_\d{2}_\d{3})\.csv")
                    file_glob = "G2 *.csv"
                else:
                    pattern = re.compile(r"Take (\d{4}-\d{2}-\d{2} \d{1,2}\.\d{2}\.\d{2} [AP]M)")
                    file_glob = "Take *.csv"

                for file in folder.glob(file_glob):
                    m = pattern.search(file.name)
                    if not m:
                        continue

                    if group_name == 'Group2':
                        dt_str = m.group(1)
                        date_part, time_part = dt_str.split('_', 1)
                        time_parts = time_part.split('_')
                        if len(time_parts) == 2:
                            hour, min_sec = time_parts
                            min = min_sec[:2]
                            sec = min_sec[2:]
                            dt_str = f"{date_part} {hour} {min} {sec}"
                        else:
                            dt_str = dt_str.replace('_', ' ')
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H %M %S")
                    else:
                        dt = datetime.strptime(m.group(1), "%Y-%m-%d %I.%M.%S %p")

                    df = pd.read_csv(file, skiprows=7)
                    if len(df) < 5:
                        continue
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
                        "group": group_name,
                        "size": size_key,
                        "direction": dir_key,
                        "datetime": dt,
                        "x": float(last["X.1"]),
                        "y": float(last["Y.1"]),
                        "theta": yaw
                    })
    return pd.DataFrame(data)

# ==================== LOAD YOUBOT COMBINED ====================
def load_youbot_combined():
    data = []
    groups = {
        'Group1': ROOT / 'Group1' / 'youBot',
        'Group2': ROOT / 'Group2' / 'actuator',
        'Group3': ROOT / 'Group3',
        'Group4': ROOT / 'Group4' / 'Group_4_YouBot_Data' / 'Group 4'
    }

    for group_name, base_path in groups.items():
        if not base_path.exists():
            print(f"Skipping {group_name}: path not found")
            continue

        for size_key in ["small", "medium", "large"]:
            for dir_key in ["left", "right", "straight"]:
                if group_name == 'Group1':
                    folder = base_path / size_key / dir_key
                elif group_name == 'Group2':
                    folder = base_path / f"{size_key}_{dir_key}" / "csv"
                elif group_name == 'Group3':
                    folder = base_path / f"{size_key}_{dir_key}" / "csv"
                elif group_name == 'Group4':
                    folder = base_path / size_key / dir_key / "csv"
                else:
                    continue

                if not folder.exists():
                    continue

                for csv_file in folder.glob("*.csv"):
                    stem = csv_file.stem
                    # Parse datetime from filename
                    time_match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", stem)
                    if not time_match:
                        continue
                    time_str = time_match.group(1)
                    try:
                        dt = datetime.strptime(time_str, "%Y_%m_%d_%H_%M_%S")
                    except:
                        continue

                    df = pd.read_csv(csv_file, header=None, names=["x","y","theta"])
                    if df.empty:
                        continue
                    last = df.iloc[-1]

                    # Standardize units: convert to cm if not already
                    x = float(last.x)
                    y = float(last.y)
                    if group_name != 'Group3':  # Group3 already in cm
                        x *= 100
                        y *= 100

                    data.append({
                        "group": group_name,
                        "size": size_key,
                        "direction": dir_key,
                        "datetime": dt,
                        "x": x,
                        "y": y,
                        "theta": float(last.theta)
                    })
    return pd.DataFrame(data)

# ==================== OUTLIER REMOVAL (IQR) ====================
def remove_outliers_per_group(df):
    cleaned_groups = []
    outlier_counts = {}

    for (group, size, direction), group_df in df.groupby(["group", "size", "direction"]):
        key = f"{group}_{size}_{direction}"
        if len(group_df) < 4:
            cleaned_groups.append(group_df)
            outlier_counts[key] = 0
            continue

        Q1 = group_df[["x_youbot", "y_youbot"]].quantile(0.25)
        Q3 = group_df[["x_youbot", "y_youbot"]].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((group_df[["x_youbot", "y_youbot"]] < (Q1 - 1.5*IQR)) |
                  (group_df[["x_youbot", "y_youbot"]] > (Q3 + 1.5*IQR))).any(axis=1)

        outliers = group_df[~mask]
        outlier_counts[key] = len(outliers)
        cleaned_groups.append(group_df[mask])

    # Print summary
    print("\nOutlier removal summary:")
    total = 0
    for key in sorted(outlier_counts):
        cnt = outlier_counts[key]
        print(f"   {key}: {cnt} outliers removed")
        total += cnt
    print(f"   TOTAL outliers removed: {total}\n")

    return pd.concat(cleaned_groups, ignore_index=True), outlier_counts

# ==================== MAIN ====================
print("Loading combined data from all groups...")
opti_df = load_optitrack_combined()
youbot_df = load_youbot_combined()
print(f"OptiTrack trials: {len(opti_df)}")
print(f"YouBot trials: {len(youbot_df)}")

# Combine all data
opti_df['type'] = 'external'
youbot_df['type'] = 'internal'
matched = pd.concat([opti_df, youbot_df], ignore_index=True)
print(f"Combined {len(matched)} trials across all groups\n")

# Skip outlier removal for combined
matched_clean = matched

# ==================== CREATE COMBINED PLOTS ====================
group_colors = {'Group1': 'blue', 'Group2': 'green', 'Group3': 'red', 'Group4': 'orange'}
markers = {'straight': 'o', 'left': '^', 'right': 's'}

for size in sorted(matched_clean['size'].unique()):
    plt.figure(figsize=(12, 10))

    for direction in ['straight', 'left', 'right']:
        for group_name in sorted(matched_clean['group'].unique()):
            # YouBot (internal)
            youbot_sub = matched_clean[(matched_clean['size'] == size) & (matched_clean['direction'] == direction) & (matched_clean['group'] == group_name) & (matched_clean['type'] == 'internal')]
            if not youbot_sub.empty:
                plt.scatter(youbot_sub.x, youbot_sub.y, c=group_colors.get(group_name, 'black'), s=50, marker=markers[direction],
                           label=f"YouBot {group_name} {direction}", edgecolors='black', linewidth=1, zorder=10, alpha=0.7)
                # Arrows for orientation
                for _, r in youbot_sub.iterrows():
                    dx = 8 * np.sin(r.theta)
                    dy = 8 * np.cos(r.theta)
                    plt.arrow(r.x, r.y, dx, dy, head_width=3, color=group_colors.get(group_name, 'black'),
                             length_includes_head=True, zorder=15, alpha=0.7)

            # OptiTrack (external)
            opti_sub = matched_clean[(matched_clean['size'] == size) & (matched_clean['direction'] == direction) & (matched_clean['group'] == group_name) & (matched_clean['type'] == 'external')]
            if not opti_sub.empty:
                plt.scatter(opti_sub.x, opti_sub.y, c=group_colors.get(group_name, 'black'), s=30, marker='X',
                           label=f"OptiTrack {group_name} {direction}", zorder=5, alpha=0.7)
                # Arrows for OptiTrack
                for _, r in opti_sub.iterrows():
                    dx = 6 * np.sin(r.theta)
                    dy = 6 * np.cos(r.theta)
                    plt.arrow(r.x, r.y, dx, dy, head_width=2, color=group_colors.get(group_name, 'black'),
                             alpha=0.7, length_includes_head=True, zorder=20)

    plt.xlabel("X [cm]", fontsize=14)
    plt.ylabel("Y – Forward direction [cm]", fontsize=14)
    plt.title(f"Final End-Effector and Object Poses\n{size.capitalize()} Object – All Directions (Combined Groups)\n"
              f"({len(matched_clean[matched_clean['size'] == size])} trials)", fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))

    filename = ROOT / f"combined_{size}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename.name}")

# Compute accuracy and precision per group
print("\nComputing statistical analysis for combined data...")

accuracy = {}
precision = {}

for (group_name, size, direction), group_df in matched_clean.groupby(["group", "size", "direction"]):
    internal = group_df[group_df['type'] == 'internal']
    external = group_df[group_df['type'] == 'external']
    print(f"Group {group_name} {size}_{direction}: {len(internal)} internal, {len(external)} external trials")
    if not internal.empty and not external.empty:
        # Accuracy: mean distance between all internal and all external
        distances = []
        for _, i in internal.iterrows():
            for _, e in external.iterrows():
                dist = np.sqrt((i.x - e.x)**2 + (i.y - e.y)**2)
                distances.append(dist)
        accuracy[f"{group_name}_{size}_{direction}"] = np.mean(distances)
        
        # Precision: mean std dev of external positions
        std_x = external["x"].std()
        std_y = external["y"].std()
        precision[f"{group_name}_{size}_{direction}"] = (std_x + std_y) / 2
    else:
        accuracy[f"{group_name}_{size}_{direction}"] = np.nan
        precision[f"{group_name}_{size}_{direction}"] = np.nan

print("\nAccuracy (mean distance between internal and external, cm):")
for key, val in sorted(accuracy.items()):
    if not np.isnan(val):
        print(f"  {key}: {val:.2f}")

print("\nPrecision (mean std dev of external positions, cm):")
for key, val in sorted(precision.items()):
    if not np.isnan(val):
        print(f"  {key}: {val:.2f}")

# Save preprocessed data as CSV
preprocessed = []
for idx, row in matched_clean.iterrows():
    preprocessed.append({
        "group": row.group,
        "mass": row.size,
        "position": row.direction,
        "trial": 1,  # Placeholder since not matched
        "type": row.type,
        "x": row.x,
        "y": row.y,
        "theta": row.theta
    })

preprocessed_df = pd.DataFrame(preprocessed)
preprocessed_df.to_csv(ROOT / "combined_preprocessed_data.csv", index=False)
print(f"\nSaved combined preprocessed data to {ROOT / 'combined_preprocessed_data.csv'}")

print("\nAll combined plots saved successfully!")