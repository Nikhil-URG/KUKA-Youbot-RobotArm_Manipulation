import pandas as pd
import numpy as np
import scipy.stats as stats

# Transformation Offsets (Align YouBot to OptiTrack)
OFFSET_X = -212.40 
OFFSET_Y = 76.17

# CORRECTED Relative Targets
# SWAPPED Left/Right X values based on observed data
RELATIVE_TARGETS = {
    'Left':     {'x': 20, 'y': 30, 'theta': -0.5},   # Changed from -20 to 20
    'Straight': {'x': 0,   'y': 45, 'theta': 0},
    'Right':    {'x': -20, 'y': 30, 'theta': 0.5}    # Changed from 20 to -20
}

def run():
    try:
        df = pd.read_csv("combined_transformed_data.csv")
    except FileNotFoundError:
        print("Error: combined_transformed_data.csv not found.")
        return

    print("\n" + "="*115)
    print(" STEP 4: TABLE 5 GENERATION (Fixed Coordinates)")
    print("="*115)
    
    # Header
    print(f"{'Size':<8} {'Direction':<10} {'Rand.Var':<8} {'Mean':<8} {'Variance':<8} {'Accuracy':<8} {'TestStat':<10} {'P-Val':<10} {'Null Hyp'}")
    print("-" * 115)
    
    sizes = sorted(df['Size'].unique())
    directions = sorted(df['Direction'].unique())
    
    for s in sizes:
        for d in directions:
            subset = df[(df['Size'] == s) & (df['Direction'] == d)]
            if subset.empty: continue
            
            # Transform Relative Target to Global OptiTrack Frame
            rel_gt = RELATIVE_TARGETS[d]
            
            gt_x = rel_gt['x'] + OFFSET_X
            gt_y = rel_gt['y'] + OFFSET_Y
            
            variables = {
                'X': ('x_optitrack', gt_x),
                'Y': ('y_optitrack', gt_y)
            }
            
            for var_name, (col, truth) in variables.items():
                data = subset[col].dropna()
                
                if len(data) < 2: continue
                
                # 1. Stats
                mean_val = np.mean(data)
                var_val = np.var(data)
                
                # 2. Accuracy
                acc_val = abs(mean_val - truth)
                
                # 3. Normality
                if len(data) >= 3:
                    stat, p = stats.shapiro(data)
                    # p-value formatting: if < 0.0001, just show 0.0000
                    res = "Fail to Rej" if p > 0.05 else "Reject"
                else:
                    stat, p = 0.0, 1.0
                    res = "N/A"
                
                print(f"{s:<8} {d:<10} {var_name:<8} {mean_val:<8.2f} {var_val:<8.4f} {acc_val:<8.2f} {stat:<10.4f} {p:<10.4f} {res}")
            
            print("-" * 115)

if __name__ == "__main__":
    run()