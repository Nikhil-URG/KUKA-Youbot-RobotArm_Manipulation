import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    # Compute theta from quaternion
    forward_local = np.array([1, 0, 0])
    forward_world = rotate_vector_by_quaternion(forward_local, quat)
    theta = np.arctan2(forward_world[2], forward_world[0])  # atan2(y, x) in the rotated frame
    return x_y, y_y, theta

# Function to process Youbot file
def process_youbot_file(filepath):
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        return None
    last_row = df.iloc[-1]
    pos = np.array([last_row.iloc[0], last_row.iloc[1], last_row.iloc[2]])
    theta = pos[2]  # 3rd column as orientation
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

# Compute accuracies before outlier removal
accuracies = {}
for obj in objects:
    for dir_ in directions:
        points = np.array(opti_data[obj][dir_])
        if len(points) > 0 and obj in youbot_data and dir_ in youbot_data[obj]:
            youbot_points = np.array(youbot_data[obj][dir_])
            if len(youbot_points) > 0:
                # Compute distance between mean positions
                opti_mean = np.mean(points[:, :2], axis=0)
                youbot_mean = np.mean(youbot_points[:, :2], axis=0)
                dist = np.linalg.norm(opti_mean - youbot_mean) * 100  # to cm
                accuracies[(obj, dir_)] = dist
                print(f"{obj} {dir_}: Opti samples={len(points)}, Youbot samples={len(youbot_points)}, Opti mean={opti_mean}, Youbot mean={youbot_mean}, Dist={dist:.3f} cm")

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

# Create gaussian directory
gauss_dir = os.path.join(base_dir, 'gaussian_analysis')
os.makedirs(gauss_dir, exist_ok=True)

# Gaussian analysis
for source, data in [('opti', opti_data), ('youbot', youbot_data)]:
    for obj in objects:
        for dir_ in directions:
            points = np.array(data[obj][dir_])
            if len(points) > 1:
                x_data = points[:, 0]
                y_data = points[:, 1]
                theta_data = points[:, 2]

                # Compute accuracy if both sources available
                accuracy = 0.0
                if source == 'opti' and obj in youbot_data and dir_ in youbot_data[obj]:
                    youbot_points = np.array(youbot_data[obj][dir_])
                    if len(youbot_points) == len(points):
                        distances = np.sqrt((points[:, 0] - youbot_points[:, 0])**2 + (points[:, 1] - youbot_points[:, 1])**2)
                        accuracy = np.mean(distances)

                # Print stats for table
                print(f'{source} {obj} {dir_}:')
                print(f'  X: Mean={np.mean(x_data):.3f}, Var={np.var(x_data):.6f}, Acc={accuracy:.3f}')
                if len(x_data) > 1:
                    stat, p_val = stats.shapiro(x_data)
                    null_hyp = 'Fail to Reject' if p_val > 0.05 else 'Reject'
                    print(f'    Shapiro p={p_val:.3f}, Null: {null_hyp}')
                print(f'  Y: Mean={np.mean(y_data):.3f}, Var={np.var(y_data):.6f}')
                if len(y_data) > 1:
                    stat, p_val = stats.shapiro(y_data)
                    null_hyp = 'Fail to Reject' if p_val > 0.05 else 'Reject'
                    print(f'    Shapiro p={p_val:.3f}, Null: {null_hyp}')
                print(f'  Theta: Mean={np.mean(theta_data):.3f}, Var={np.var(theta_data):.6f}')
                if len(theta_data) > 1:
                    stat, p_val = stats.shapiro(theta_data)
                    null_hyp = 'Fail to Reject' if p_val > 0.05 else 'Reject'
                    print(f'    Shapiro p={p_val:.3f}, Null: {null_hyp}')

                # X distribution
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.hist(x_data, bins=20, alpha=0.7, density=True, label='Data')
                if len(x_data) > 1:
                    mu_x, std_x = stats.norm.fit(x_data)
                    xmin, xmax = plt.xlim()
                    x_fit = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x_fit, mu_x, std_x)
                    plt.plot(x_fit, p, 'k', linewidth=2, label=f'Gaussian fit\nμ={mu_x:.3f}, σ={std_x:.3f}')
                    # Shapiro-Wilk test
                    stat, p_value = stats.shapiro(x_data)
                    plt.text(0.05, 0.95, f'Shapiro-Wilk p={p_value:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
                plt.title(f'{source.capitalize()} {obj} {dir_} - X Position')
                plt.xlabel('X (m)')
                plt.ylabel('Density')
                plt.legend()

                # Y distribution
                plt.subplot(1, 2, 2)
                plt.hist(y_data, bins=20, alpha=0.7, density=True, label='Data')
                if len(y_data) > 1:
                    mu_y, std_y = stats.norm.fit(y_data)
                    ymin, ymax = plt.ylim()
                    y_fit = np.linspace(ymin, ymax, 100)
                    p = stats.norm.pdf(y_fit, mu_y, std_y)
                    plt.plot(y_fit, p, 'k', linewidth=2, label=f'Gaussian fit\nμ={mu_y:.3f}, σ={std_y:.3f}')
                    stat, p_value = stats.shapiro(y_data)
                    plt.text(0.05, 0.95, f'Shapiro-Wilk p={p_value:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
                plt.title(f'{source.capitalize()} {obj} {dir_} - Y Position')
                plt.xlabel('Y (m)')
                plt.ylabel('Density')
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(gauss_dir, f'{source}_{obj}_{dir_}_gaussian.png'))
                plt.close()

print('Gaussian analysis plots saved in gaussian_analysis/ folder')