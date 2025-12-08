import pandas as pd
import numpy as np
import os
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_MOCK_DATA = False 

OFFSET_X = -212.40 
OFFSET_Y = 76.17

GROUND_TRUTH = {
    'Left':     {'x': -20, 'y': 30, 'theta': -0.5},
    'Straight': {'x': 0,   'y': 45, 'theta': 0},
    'Right':    {'x': 20,  'y': 30, 'theta': 0.5}
}

# =============================================================================
# ROBUST PARSING FUNCTIONS
# =============================================================================
def parse_youbot_csv(filepath):
    """
    Parses YouBot CSV. Format: No header, x, y, theta.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        if df.empty: return None
        
        last_row = df.iloc[-1]
        x_cm = last_row[0] * 100
        y_cm = last_row[1] * 100
        theta = last_row[2]
        return x_cm, y_cm, theta
    except Exception as e:
        print(f"  [Error] YouBot Parse ({os.path.basename(filepath)}): {e}")
        return None

def parse_optitrack_csv(filepath):
    """
    Parses OptiTrack CSV (Motive Export).
    Robustly handles metadata headers and duplicate X/Y/Z columns (Rotation vs Position).
    """
    try:
        # 1. Detect Header Row dynamically
        # We search for the line containing specific keywords that indicate the column names
        header_row_idx = None
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                # Your files typically have ",Rotation X, ... ,Position X," in the header
                if "Position" in line and "Rotation" in line:
                    header_row_idx = i
                    break
                # Fallback: sometimes it starts with Frame
                if line.startswith("Frame,") and "Time" in line:
                    header_row_idx = i
                    break
        
        if header_row_idx is None:
            # Fallback: Assume row 6 if search fails (common in Motive 1.x)
            header_row_idx = 6

        # 2. Read CSV
        # We skip the row immediately after the header because it often contains units or is empty
        # which confuses pandas type inference.
        df = pd.read_csv(filepath, header=header_row_idx)
        
        # 3. Find Position Columns
        # Motive exports Rotation (X,Y,Z,W) FIRST, then Position (X,Y,Z).
        # Pandas handles duplicate names by adding suffixes (X, X.1).
        # Strategy: Find ALL columns starting with "X" or "Position X" and take the LAST ones.
        
        cols = [str(c).strip() for c in df.columns]
        
        # Filter for columns that look like X or Z (OptiTrack Z = World Y)
        # We look for "Position X" or just "X" if the prefix isn't there
        x_candidates = [i for i, c in enumerate(cols) if "Position X" in c or c == "X" or c == "X.1"]
        z_candidates = [i for i, c in enumerate(cols) if "Position Z" in c or c == "Z" or c == "Z.1"]
        
        if not x_candidates or not z_candidates:
            # Fallback for some rigid body exports: Look for ".X" and ".Z"
            x_candidates = [i for i, c in enumerate(cols) if ".X" in c]
            z_candidates = [i for i, c in enumerate(cols) if ".Z" in c]

        if not x_candidates or not z_candidates:
            return None

        # The Position columns are invariably the LAST set in standard Rigid Body exports
        x_idx = x_candidates[-1]
        z_idx = z_candidates[-1]

        # 4. Get Final Data Point (Last valid row)
        # Drop rows that might be completely empty at the end
        df = df.dropna(subset=[df.columns[x_idx]])
        
        if df.empty: return None
        
        last_row = df.iloc[-1]
        
        # Force convert to float (handle any weird string formatting)
        raw_x = float(last_row.iloc[x_idx])
        raw_z = float(last_row.iloc[z_idx]) 
        
        # 5. Unit Conversion
        # If value is > 500 (e.g. 2000 mm), convert to cm.
        if abs(raw_x) > 500: 
            return raw_x / 10.0, raw_z / 10.0
        return raw_x, raw_z

    except Exception as e:
        print(f"  [Error] OptiTrack Parse ({os.path.basename(filepath)}): {e}")
        return None

# =============================================================================
# DATA LOADER
# =============================================================================
def load_and_transform_data():
    print(f"--- SCANNING FOR DATA ---")
    groups = ['Group1', 'Group2', 'Group3', 'Group4']
    found_files = []

    # 1. CRAWL DIRECTORIES
    for root, dirs, files in os.walk("."):
        if not any(g in root for g in groups): continue
        
        for file in files:
            if not file.endswith(".csv"): continue
            if file == "combined_transformed_data.csv": continue
            
            full_path = os.path.join(root, file)
            lower_path = full_path.lower()
            
            # Attributes
            grp = next((g for g in groups if g in root), "Unknown")
            
            size = "Unknown"
            if "small" in lower_path: size = "Small"
            elif "medium" in lower_path: size = "Medium"
            elif "large" in lower_path: size = "Large"
            
            direction = "Unknown"
            if "left" in lower_path: direction = "Left"
            elif "right" in lower_path: direction = "Right"
            elif "straight" in lower_path: direction = "Straight"
            
            sensor = "Unknown"
            if "youbot" in lower_path or file.startswith("2025"): sensor = "youBot"
            elif "optitrack" in lower_path or "take" in lower_path or file.startswith("Take") or file.startswith("G"): sensor = "OptiTrack"

            if size != "Unknown" and direction != "Unknown" and sensor != "Unknown":
                found_files.append({'Path': full_path, 'Group': grp, 'Size': size, 'Direction': direction, 'Sensor': sensor})

    # 2. MATCH PAIRS
    data_rows = []
    unique_keys = set((f['Group'], f['Size'], f['Direction']) for f in found_files)
    
    print(f"Found {len(found_files)} candidate files. Matching pairs...")

    for (g, s, d) in unique_keys:
        ybs = sorted([f['Path'] for f in found_files if f['Group']==g and f['Size']==s and f['Direction']==d and f['Sensor']=='youBot'])
        opts = sorted([f['Path'] for f in found_files if f['Group']==g and f['Size']==s and f['Direction']==d and f['Sensor']=='OptiTrack'])
        
        n_pairs = min(len(ybs), len(opts))
        
        for i in range(n_pairs):
            yb = parse_youbot_csv(ybs[i])
            opt = parse_optitrack_csv(opts[i])
            
            if yb and opt:
                # TRANSFORM
                yb_x_trans = yb[0] + OFFSET_X
                yb_y_trans = yb[1] + OFFSET_Y
                acc_err = np.sqrt((yb_x_trans - opt[0])**2 + (yb_y_trans - opt[1])**2)
                
                data_rows.append({
                    'Group': g, 'Size': s, 'Direction': d,
                    'x_youbot_transformed': yb_x_trans,
                    'y_youbot_transformed': yb_y_trans,
                    'theta': yb[2],
                    'x_optitrack': opt[0], 'y_optitrack': opt[1],
                    'accuracy_error': acc_err
                })
            else:
                # Print errors only for the pairs we attempted to match
                if not yb: print(f"  Failed YouBot: {os.path.basename(ybs[i])}")
                if not opt: print(f"  Failed OptiTrack: {os.path.basename(opts[i])}")

    if not data_rows:
        print("ERROR: No matching data rows created. Check parsing errors above.")
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)
    print(f"SUCCESS: Loaded {len(df)} valid trials.")
    return df

if __name__ == "__main__":
    df = load_and_transform_data()
    if not df.empty:
        df.to_csv("combined_transformed_data.csv", index=False)
        print("Data exported to combined_transformed_data.csv")