import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base_dir = '.'

# Offset for OptiTrack
offset_opti = np.array([-212.40, 76.17, 103.15])

# Function to rotate vector by quaternion
def rotate_vector_by_quaternion(v, q):
    # q = [w, x, y, z]
    w, x, y, z = q
    # Compute v' = q * v * q^-1
    q_inv = np.array([w, -x, -y, -z])
    # First, v as quaternion [0, vx, vy, vz]
    vq = np.array([0, v[0], v[1], v[2]])
    # q * vq
    temp = np.array([
        q[0]*vq[0] - q[1]*vq[1] - q[2]*vq[2] - q[3]*vq[3],
        q[0]*vq[1] + q[1]*vq[0] + q[2]*vq[3] - q[3]*vq[2],
        q[0]*vq[2] - q[1]*vq[3] + q[2]*vq[0] + q[3]*vq[1],
        q[0]*vq[3] + q[1]*vq[2] - q[2]*vq[1] + q[3]*vq[0]
    ])
    # temp * q_inv
    result = np.array([
        temp[0]*q_inv[0] - temp[1]*q_inv[1] - temp[2]*q_inv[2] - temp[3]*q_inv[3],
        temp[0]*q_inv[1] + temp[1]*q_inv[0] + temp[2]*q_inv[3] - temp[3]*q_inv[2],
        temp[0]*q_inv[2] - temp[1]*q_inv[3] + temp[2]*q_inv[0] + temp[3]*q_inv[1],
        temp[0]*q_inv[3] + temp[1]*q_inv[2] - temp[2]*q_inv[1] + temp[3]*q_inv[0]
    ])
    return result[1:4]  # Return vector part

# Function to process OptiTrack file
def process_opti_file(filepath):
    df = pd.read_csv(filepath, skiprows=8)  # Skip headers
    if df.empty:
        return None
    last_row = df.iloc[-1]
    pos = np.array([last_row.iloc[6], last_row.iloc[7], last_row.iloc[8]])  # X,Y,Z position
    quat = np.array([last_row.iloc[3], last_row.iloc[4], last_row.iloc[5], last_row.iloc[6]])  # W,X,Y,Z
    pos_robot = pos - offset_opti
    # Rotate: x_y = x, y_y = z, z_y = y
    x_y = pos_robot[0] / 100  # to meters
    y_y = pos_robot[2] / 100
    z_y = pos_robot[1] / 100
    # Compute orientation: rotate [1,0,0] by quaternion, then apply same rotation
    forward_local = np.array([1, 0, 0])
    forward_world = rotate_vector_by_quaternion(forward_local, quat)
    # Then apply the axis swap: x->x, y->z, z->y
    arrow_length = 0.02  # 1cm
    norm_xy = np.linalg.norm(forward_world[:2])
    if norm_xy > 0:
        dir_x = (forward_world[0] / norm_xy) * arrow_length
        dir_y = (forward_world[2] / norm_xy) * arrow_length
    else:
        dir_x = 0
        dir_y = 0
    return x_y, y_y, z_y, dir_x, dir_y

# Function to process Youbot file
def process_youbot_file(filepath):
    df = pd.read_csv(filepath, header=None)
    if df.empty:
        return None
    last_row = df.iloc[-1]
    pos = np.array([last_row.iloc[0], last_row.iloc[1], last_row.iloc[2]])  # keep in meters
    yaw = pos[2]  # 3rd column as orientation (yaw)
    arrow_length = 0.02  # 1cm
    dir_x = np.cos(yaw) * arrow_length
    dir_y = np.sin(yaw) * arrow_length
    return pos[0], pos[1], pos[2], dir_x, dir_y

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
                # Try Group2 style: obj_dir
                dir_name = f"{obj}_{dir_}"
                dir_path = os.path.join(opti_dir, dir_name)
                if os.path.exists(dir_path):
                    print(f"Processing opti {group} {obj} {dir_} in {dir_path}")
                    for file in os.listdir(dir_path):
                        if file.endswith('.csv'):
                            filepath = os.path.join(dir_path, file)
                            point = process_opti_file(filepath)
                            if point:
                                opti_data[obj][dir_].append(point)
                else:
                    # Try Group1 style: obj/dir_
                    obj_dir = os.path.join(opti_dir, obj)
                    if os.path.exists(obj_dir):
                        dir_path = os.path.join(obj_dir, dir_)
                        if os.path.exists(dir_path):
                            for file in os.listdir(dir_path):
                                if file.endswith('.csv'):
                                    filepath = os.path.join(dir_path, file)
                                    point = process_opti_file(filepath)
                                    if point:
                                        opti_data[obj][dir_].append(point)  # point is (x,y,z,dx,dy)

    # Youbot
    youbot_dir = os.path.join(group_dir, 'youbot')  # or 'youBot' for Group1
    if not os.path.exists(youbot_dir):
        youbot_dir = os.path.join(group_dir, 'youBot')
    if os.path.exists(youbot_dir):
        for obj in objects:
            for dir_ in directions:
                # Try Group2 style: obj_dir/csv
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
                    # Try Group1 style: obj/dir_/csv
                    obj_dir = os.path.join(youbot_dir, obj)
                    if os.path.exists(obj_dir):
                        dir_path = os.path.join(obj_dir, dir_, 'csv')
                        if os.path.exists(dir_path):
                            for file in os.listdir(dir_path):
                                if file.endswith('.csv'):
                                    filepath = os.path.join(dir_path, file)
                                    point = process_youbot_file(filepath)
                                    if point:
                                        youbot_data[obj][dir_].append(point)
        
        # Outlier removal for OptiTrack data
        for obj in objects:
            for dir_ in directions:
                points = np.array(opti_data[obj][dir_])
                if len(points) > 0:
                    # Remove outliers based on IQR for x and y
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
        
        # Create plots directory
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Generate plots
for obj in objects:
    # Opti plot
    plt.figure()
    for dir_ in directions:
        points = np.array(opti_data[obj][dir_])
        if points.size:
            plt.scatter(points[:, 0], points[:, 1], label=dir_, alpha=0.7)
            # Add arrows for orientation
            plt.quiver(points[:, 0], points[:, 1], points[:, 3], points[:, 4], angles='xy', scale_units='xy', scale=1, width=0.001, alpha=0.7)
    plt.title(f'OptiTrack {obj}')
    plt.xlabel('Position in the X-axis direction (m)')
    plt.ylabel('Position in the Y-axis direction (m)')
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, f'optitrack_{obj}.png'))
    plt.close()

    # Youbot plot
    plt.figure()
    for dir_ in directions:
        points = np.array(youbot_data[obj][dir_])
        if points.size:
            plt.scatter(points[:, 0], points[:, 1], label=dir_, alpha=0.7)
            plt.quiver(points[:, 0], points[:, 1], points[:, 3], points[:, 4], angles='xy', scale_units='xy', scale=1, width=0.001, alpha=0.7)
    plt.title(f'Youbot {obj}')
    plt.xlabel('Position in the X-axis direction (m)')
    plt.ylabel('Position in the Y-axis direction (m)')
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, f'youbot_{obj}.png'))
    plt.close()

    # Combined plot
    plt.figure()
    for dir_ in directions:
        opti_points = np.array(opti_data[obj][dir_])
        youbot_points = np.array(youbot_data[obj][dir_])
        if opti_points.size:
            plt.scatter(opti_points[:, 0], opti_points[:, 1], marker='o', label=f'{dir_} OptiTrack', alpha=0.7)
            plt.quiver(opti_points[:, 0], opti_points[:, 1], opti_points[:, 3], opti_points[:, 4], angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.7)
        if youbot_points.size:
            plt.scatter(youbot_points[:, 0], youbot_points[:, 1], marker='s', label=f'{dir_} Youbot', alpha=0.7)
            plt.quiver(youbot_points[:, 0], youbot_points[:, 1], youbot_points[:, 3], youbot_points[:, 4], angles='xy', scale_units='xy', scale=1, width=0.001, alpha=0.7)
    plt.title(f'Combined {obj}')
    plt.xlabel('Position in the X-axis direction (m)')
    plt.ylabel('Position in the Y-axis direction (m)')
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, f'combined_{obj}.png'))
    plt.close()

# Print counts
for group in groups:
    opti_count = sum(len(opti_data[obj][dir_]) for obj in objects for dir_ in directions)
    youbot_count = sum(len(youbot_data[obj][dir_]) for obj in objects for dir_ in directions)
    print(f'{group}: Opti {opti_count}, Youbot {youbot_count}')

print('Total Opti points:', sum(len(opti_data[obj][dir_]) for obj in objects for dir_ in directions))
print('Total Youbot points:', sum(len(youbot_data[obj][dir_]) for obj in objects for dir_ in directions))

# Statistical analysis
print('\nStatistical Analysis (Mean and Std for X, Y positions in meters):')
for obj in objects:
    for dir_ in directions:
        opti_points = np.array(opti_data[obj][dir_])
        youbot_points = np.array(youbot_data[obj][dir_])
        if opti_points.size:
            print(f'Opti {obj} {dir_}: Mean X={opti_points[:, 0].mean():.2f}, Y={opti_points[:, 1].mean():.2f}, Std X={opti_points[:, 0].std():.2f}, Y={opti_points[:, 1].std():.2f}')
        if youbot_points.size:
            print(f'Youbot {obj} {dir_}: Mean X={youbot_points[:, 0].mean():.2f}, Y={youbot_points[:, 1].mean():.2f}, Std X={youbot_points[:, 0].std():.2f}, Y={youbot_points[:, 1].std():.2f}')