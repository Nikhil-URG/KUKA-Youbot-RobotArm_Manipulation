import rosbag
import numpy as np
import sys
import os
import glob

def get_poses(bagfile):
    bag = rosbag.Bag(bagfile)
    poses = []
    try:
        for topic, msg, t in bag.read_messages(topics=['/end_effector_pose']):
            poses.append([msg.x, msg.y, msg.theta])

    except:
        print('ignoring %s' % bagfile)
        bag.close()
    	return None

    bag.close()
    poses = np.array(poses)
    print(poses)
    return poses

def main():
    if len(sys.argv) < 2:
        print("specify root directory")
        return
    root_dir = str(sys.argv[1])
    csv_path = os.path.join(root_dir, 'csv')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    files = sorted(glob.glob(root_dir + '/*.bag'))
    for f in files:
        filename = os.path.basename(f)[:-4] + '.csv'
        data = get_poses(f)
	if data is not None:
		np.savetxt(os.path.join(csv_path, filename), data, delimiter=',')

if __name__ == '__main__':
    main()
