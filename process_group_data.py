import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
OFFSET_X = -212.40 
OFFSET_Y = 76.17
OUTPUT_FILE = "combined_transformed_data.csv"

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================
def parse_youbot_csv(filepath):
    try:
        # YouBot: No header, x(m), y(m), theta(rad)
        df = pd.read_csv(filepath, header=None)
        if df.empty: return None
        last_row = df.iloc[-1]
        return last_row[0] * 100, last_row[1] * 100, last_row[2] # Convert m -> cm
    except: return None

def parse_optitrack_csv(filepath):
    try:
        # OptiTrack: Metadata header, find Position X/Z
        header_idx = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith("Frame,"):
                    header_idx = i
                    break
        
        df = pd.read_csv(filepath, header=header_idx, skiprows=[header_idx+1])
        
        # Find columns (OptiTrack Z maps to World Y)
        cols = df.columns
        x_col = next((c for c in cols if "Position.X" in c), None)
        z_col = next((c for c in cols if "Position.Z" in c), None)
        
        if not x_col or not z_col: return None # Fallback failed
        
        last_row = df.iloc[-1]
        rx, rz = last_row[x_col], last_row[z_col]
        
        # Unit check (if >500, likely mm, convert to cm)
        if abs(rx) > 500: return rx/10.0, rz/10.0
        return rx, rz
    except: return None

# =============================================================================
# SMART CRAWLER & MERGER
# =============================================================================
def main():
    print(f"--- SCANNING FOR DATA (Smart Mode) ---")
    groups = ['Group1', 'Group2', 'Group3', 'Group4']
    found_files = []

    # 1. Crawl all folders
    for root, dirs, files in os.walk("."):
        # Only process relevant group folders
        if not any(g in root for g in groups): continue
        
        for file in files:
            if not file.endswith(".csv"): continue
            full_path = os.path.join(root, file)
            lower_path = full_path.lower()
            
            # Detect Attributes from File Path/Name
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
            elif "optitrack" in lower_path or "take" in lower_path or file.startswith("Take"): sensor = "OptiTrack"

            if size != "Unknown" and direction != "Unknown" and sensor != "Unknown":
                found_files.append({'Path': full_path, 'Group': grp, 'Size': size, 'Direction': direction, 'Sensor': sensor})

    # 2. Match Pairs
    data_rows = []
    unique_keys = set((f['Group'], f['Size'], f['Direction']) for f in found_files)
    
    print(f"Found {len(found_files)} candidate files. Matching pairs...")

    for (g, s, d) in unique_keys:
        ybs = sorted([f['Path'] for f in found_files if f['Group']==g and f['Size']==s and f['Direction']==d and f['Sensor']=='youBot'])
        opts = sorted([f['Path'] for f in found_files if f['Group']==g and f['Size']==s and f['Direction']==d and f['Sensor']=='OptiTrack'])
        
        for i in range(min(len(ybs), len(opts))):
            yb = parse_youbot_csv(ybs[i])
            opt = parse_optitrack_csv(opts[i])
            
            if yb and opt:
                # 3. Transform
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

    if not data_rows:
        print("ERROR: No matching data found.")
        return

    # 4. Save
    df = pd.DataFrame(data_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Saved {len(df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()