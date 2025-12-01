# convert_all_bags_to_csv_WORKING.py
# Tested & working on macOS Apple Silicon + Python 3.12 + LZ4 bags (Dec 2025)

from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.serde import ros1_to_cdr
import pandas as pd
from tqdm import tqdm

# ==============================
# CHANGE ONLY THESE TWO LINES
# ==============================
INPUT_FOLDER  = "/Users/nikhilravi/Library/CloudStorage/GoogleDrive-nikhil.urg@gmail.com/My Drive/SEE/EXP5/Youbot_Group3_WS25/Group3/large_left"
OUTPUT_FOLDER = "/Users/nikhilravi/Library/CloudStorage/GoogleDrive-nikhil.urg@gmail.com/My Drive/SEE/EXP5/Youbot_Group3_WS25/KUKA_csv/"
# ==============================

def flatten(msg, prefix=""):
    """Flatten any ROS message (including lists and nested messages)"""
    flat = {}
    if hasattr(msg, "__slots__"):
        for slot_names in msg.__slots__:
            for slot in slot_names:
                value = getattr(msg, slot)
                new_prefix = f"{prefix}.{slot}" if prefix else slot
                if hasattr(value, "__slots__") or isinstance(value, (list, tuple)):
                    flat.update(flatten(value, new_prefix=new_prefix))
                else:
                    flat[new_prefix] = value
    elif isinstance(msg, (list, tuple)):
        for i, item in enumerate(msg):
            flat.update(flatten(item, new_prefix=f"{prefix}[{i}]"))
    else:
        flat[prefix] = msg
    return flat

def bag_to_csv(bag_path: Path, out_base: Path):
    print(f"\nProcessing → {bag_path.name}")

    try:
        with Reader(bag_path) as reader:          # <-- Path object works directly
            bag_out = out_base / bag_path.stem
            bag_out.mkdir(parents=True, exist_ok=True)

            data = {}

            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                if topic not in data:
                    data[topic] = []

                # Convert ROS1 → CDR → Python object
                cdr_data = ros1_to_cdr(rawdata, connection.msgtype)
                msg = reader.deserialize(cdr_data, connection.msgtype)

                flat_dict = flatten(msg)
                flat_dict["timestamp"] = timestamp
                flat_dict["time_sec"] = timestamp / 1e9

                data[topic].append(flat_dict)

            # Save to CSV
            csv_count = 0
            for topic, messages in data.items():
                if not messages:
                    continue
                df = pd.DataFrame(messages)
                df = df.sort_values("timestamp").reset_index(drop=True)

                safe_name = topic.replace("/", "_").lstrip("_")
                csv_path = bag_out / f"{safe_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"   → {topic} ({len(df)} messages) → {csv_path.name}")
                csv_count += 1

            print(f"Done {bag_path.name} → {csv_count} CSV files created")

    except Exception as e:
        print(f"   ERROR with {bag_path.name}: {e}")

def main():
    in_path = Path(INPUT_FOLDER).expanduser().resolve()
    out_path = Path(OUTPUT_FOLDER).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    bag_files = list(in_path.rglob("*.bag"))

    if not bag_files:
        print("No .bag files found! Check INPUT_FOLDER path.")
        return

    print(f"Found {len(bag_files)} bag files. Starting conversion...\n")

    for bag in tqdm(bag_files):
        bag_to_csv(bag, out_path)

    print("\nALL DONE! CSVs are here:", out_path)

if __name__ == "__main__":
    main()