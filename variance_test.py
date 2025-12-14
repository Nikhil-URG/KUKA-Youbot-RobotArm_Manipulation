import os
import pandas as pd
import numpy as np
from scipy import stats

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

# Variance tests for OptiTrack data
print("Homogeneity of Variance Tests for OptiTrack Data")
print("=" * 50)

# Test across directions for each size
for obj in objects:
    x_groups = []
    y_groups = []
    group_labels = []
    for dir_ in directions:
        points = np.array(opti_data[obj][dir_])
        if len(points) > 1:
            x_groups.append(points[:, 0])
            y_groups.append(points[:, 1])
            group_labels.append(dir_)

    if len(x_groups) > 1:
        # Levene test for X
        stat_x, p_x = stats.levene(*x_groups)
        print(f'{obj.capitalize()} Object - X positions across directions:')
        print(f'  Levene statistic: {stat_x:.3f}, p-value: {p_x:.3f}')
        print(f'  Null hypothesis: {"Fail to Reject" if p_x > 0.05 else "Reject"} (variances equal)')
        
        # Levene test for Y
        stat_y, p_y = stats.levene(*y_groups)
        print(f'{obj.capitalize()} Object - Y positions across directions:')
        print(f'  Levene statistic: {stat_y:.3f}, p-value: {p_y:.3f}')
        print(f'  Null hypothesis: {"Fail to Reject" if p_y > 0.05 else "Reject"} (variances equal)')
        print()

# Test across sizes for each direction
for dir_ in directions:
    x_groups = []
    y_groups = []
    group_labels = []
    for obj in objects:
        points = np.array(opti_data[obj][dir_])
        if len(points) > 1:
            x_groups.append(points[:, 0])
            y_groups.append(points[:, 1])
            group_labels.append(obj)

    if len(x_groups) > 1:
        # Levene test for X
        stat_x, p_x = stats.levene(*x_groups)
        print(f'{dir_.capitalize()} Direction - X positions across sizes:')
        print(f'  Levene statistic: {stat_x:.3f}, p-value: {p_x:.3f}')
        print(f'  Null hypothesis: {"Fail to Reject" if p_x > 0.05 else "Reject"} (variances equal)')
        
        # Levene test for Y
        stat_y, p_y = stats.levene(*y_groups)
        print(f'{dir_.capitalize()} Direction - Y positions across sizes:')
        print(f'  Levene statistic: {stat_y:.3f}, p-value: {p_y:.3f}')
        print(f'  Null hypothesis: {"Fail to Reject" if p_y > 0.05 else "Reject"} (variances equal)')
        print()

print("Variance test analysis completed")