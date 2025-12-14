import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Base directory
base_dir = '.'

# Offset for OptiTrack
offset_opti = np.array([-212.40, 76.17, 103.15])

# Function to rotate vector by quaternion
def rotate_vector_by_quaternion(v, q):
    w, x, y, z = q
    q_inv = np.array([w, -x, -y, -z])
    vq = np.array([0, v[0], v[1], v[2]])
    temp = np.array([
        q[0]*vq[0] - q[1]*vq[1] - q[2]*vq[2] - q[3]*vq[3],
        q[0]*vq[1] + q[1]*vq[0] + q[2]*vq[3] - q[3]*vq[2],
        q[0]*vq[2] - q[1]*vq[3] + q[2]*vq[0] + q[3]*vq[1],
        q[0]*vq[3] + q[1]*vq[2] - q[2]*vq[1] + q[3]*vq[0]
    ])
    result = np.array([
        temp[0]*q_inv[0] - temp[1]*q_inv[1] - temp[2]*q_inv[2] - temp[3]*q_inv[3],
        temp[0]*q_inv[1] + temp[1]*q_inv[0] + temp[2]*q_inv[3] - temp[3]*q_inv[2],
        temp[0]*q_inv[2] - temp[1]*q_inv[3] + temp[2]*q_inv[0] + temp[3]*q_inv[1],
        temp[0]*q_inv[3] + temp[1]*q_inv[2] - temp[2]*q_inv[1] + temp[3]*q_inv[0]
    ])
    return result[1:4]

# Function to process OptiTrack file
def process_opti_file(filepath):
    df = pd.read_csv(filepath, skiprows=8)
    if df.empty:
        return None
    last_row = df.iloc[-1]
    pos = np.array([last_row.iloc[6], last_row.iloc[7], last_row.iloc[8]])
    quat = np.array([last_row.iloc[5], last_row.iloc[2], last_row.iloc[3], last_row.iloc[4]])  # W,X,Y,Z
    pos_robot = pos - offset_opti
    x_y = pos_robot[0] / 100
    y_y = pos_robot[2] / 100
    forward_local = np.array([1, 0, 0])
    forward_world = rotate_vector_by_quaternion(forward_local, quat)
    theta = np.arctan2(forward_world[2], forward_world[0])
    return x_y, y_y, theta

# Function to process Youbot file
def process_youbot_file(filepath):
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        return None
    last_row = df.iloc[-1]
    pos = np.array([last_row.iloc[0], last_row.iloc[1], last_row.iloc[2]])
    theta = pos[2]
    return pos[0], pos[1], theta

# Data structures
opti_data = {'small': {'left': [], 'straight': [], 'right': []},
             'medium': {'left': [], 'straight': [], 'right': []},
             'large': {'left': [], 'straight': [], 'right': []}}

youbot_data = {'small': {'left': [], 'straight': [], 'right': []},
               'medium': {'left': [], 'straight': [], 'right': []},
               'large': {'left': [], 'straight': [], 'right': []}}

groups = ['Group1', 'Group2', 'Group3', 'Group4']
objects = ['small', 'medium', 'large']
directions = ['left', 'straight', 'right']

for group in groups:
    group_dir = os.path.join(base_dir, group)
    if not os.path.exists(group_dir):
        continue

    # OptiTrack
    opti_dir = os.path.join(group_dir, 'optitrack')
    if os.path.exists(opti_dir):
        for obj in objects:
            for dir_ in directions:
                dir_name = f"{obj}_{dir_}"
                dir_path = os.path.join(opti_dir, dir_name)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith('.csv'):
                            filepath = os.path.join(dir_path, file)
                            point = process_opti_file(filepath)
                            if point:
                                opti_data[obj][dir_].append(point)
                else:
                    for cap in [obj, obj.capitalize()]:
                        obj_dir = os.path.join(opti_dir, cap)
                        if os.path.exists(obj_dir):
                            dir_path = os.path.join(obj_dir, dir_)
                            if os.path.exists(dir_path):
                                for file in os.listdir(dir_path):
                                    if file.endswith('.csv'):
                                        filepath = os.path.join(dir_path, file)
                                        point = process_opti_file(filepath)
                                        if point:
                                            opti_data[obj][dir_].append(point)
                                break

    # Youbot
    youbot_dir = os.path.join(group_dir, 'youbot')
    if not os.path.exists(youbot_dir):
        youbot_dir = os.path.join(group_dir, 'youBot')
    if os.path.exists(youbot_dir):
        for obj in objects:
            for dir_ in directions:
                dir_name = f"{obj}_{dir_}"
                dir_path = os.path.join(youbot_dir, dir_name, 'csv')
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith('.csv'):
                            filepath = os.path.join(dir_path, file)
                            point = process_youbot_file(filepath)
                            if point:
                                youbot_data[obj][dir_].append(point)
                else:
                    for cap in [obj, obj.capitalize()]:
                        obj_dir = os.path.join(youbot_dir, cap)
                        if os.path.exists(obj_dir):
                            dir_path = os.path.join(obj_dir, dir_, 'csv')
                            if os.path.exists(dir_path):
                                for file in os.listdir(dir_path):
                                    if file.endswith('.csv'):
                                        filepath = os.path.join(dir_path, file)
                                        point = process_youbot_file(filepath)
                                        if point:
                                            youbot_data[obj][dir_].append(point)
                                break

# Outlier removal for OptiTrack data
for obj in objects:
    for dir_ in directions:
        points = np.array(opti_data[obj][dir_])
        if len(points) > 0:
            x_data = points[:, 0]
            y_data = points[:, 1]
            x_q1, x_q3 = np.percentile(x_data, [25, 75])
            y_q1, y_q3 = np.percentile(y_data, [25, 75])
            x_iqr = x_q3 - x_q1
            y_iqr = y_q3 - y_q1
            x_lower = x_q1 - 1.5 * x_iqr
            x_upper = x_q3 + 1.5 * x_iqr
            y_lower = y_q1 - 1.5 * y_iqr
            y_upper = y_q3 + 1.5 * y_iqr
            mask = (x_data >= x_lower) & (x_data <= x_upper) & (y_data >= y_lower) & (y_data <= y_upper)
            opti_data[obj][dir_] = points[mask].tolist()

# Create pca directory
pca_dir = os.path.join(base_dir, 'pca_analysis')
os.makedirs(pca_dir, exist_ok=True)

# PCA analysis for OptiTrack data
for obj in objects:
    # Collect all data for the object
    all_points = []
    labels = []
    for dir_ in directions:
        points = np.array(opti_data[obj][dir_])
        if len(points) > 0:
            all_points.extend(points[:, :2])  # x, y
            labels.extend([dir_] * len(points))

    if len(all_points) > 1:
        all_points = np.array(all_points)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(all_points)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Plot
        plt.figure(figsize=(10, 8))
        colors = {'left': 'red', 'straight': 'blue', 'right': 'green'}
        for i, point in enumerate(pca_result):
            plt.scatter(point[0], point[1], color=colors[labels[i]], alpha=0.7, label=labels[i] if i < 3 else "")
        plt.title(f'PCA of OptiTrack Data for {obj.capitalize()} Object')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(pca_dir, f'pca_optitrack_{obj}.png'))
        plt.close()

        print(f'PCA for {obj}: Explained variance PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%')

print('PCA analysis plots saved in pca_analysis/ folder')