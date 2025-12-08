import pandas as pd
import scipy.stats as stats

def run():
    try:
        df = pd.read_csv("combined_transformed_data.csv")
    except FileNotFoundError:
        print("Error: combined_transformed_data.csv not found. Run common_data_loader.py first.")
        return

    print("\n" + "="*60)
    print(" STEP 3: STATISTICAL SIGNIFICANCE (Levene's Test)")
    print("="*60)
    print("Hypothesis: Variances (Precision) are equal across groups.")
    print("Variable used: 'x_optitrack' (Physical Robot Position)\n")
    
    # -------------------------------------------------------
    # 1. Effect of OBJECT SIZE on Precision (Variance)
    # -------------------------------------------------------
    # We group the data by Size: [Small_Data, Medium_Data, Large_Data]
    groups_size = []
    labels_size = sorted(df['Size'].unique())
    
    for s in labels_size:
        # We use x_optitrack to measure the REAL physical spread
        data = df[df['Size'] == s]['x_optitrack'].dropna()
        groups_size.append(data)
    
    stat_s, p_s = stats.levene(*groups_size)
    
    print(f"1. Effect of OBJECT SIZE ({', '.join(labels_size)}):")
    print(f"   Levene Statistic = {stat_s:.4f}")
    print(f"   P-value          = {p_s:.4e}")
    
    if p_s < 0.05: 
        print("   -> RESULT: SIGNIFICANT. Changing object size AFFECTS precision.")
    else: 
        print("   -> RESULT: NOT Significant. Precision is consistent across sizes.")

    print("-" * 50)

    # -------------------------------------------------------
    # 2. Effect of DIRECTION (Pose) on Precision
    # -------------------------------------------------------
    groups_dir = []
    labels_dir = sorted(df['Direction'].unique())
    
    for d in labels_dir:
        data = df[df['Direction'] == d]['x_optitrack'].dropna()
        groups_dir.append(data)
        
    stat_d, p_d = stats.levene(*groups_dir)
    
    print(f"2. Effect of PLACING POSE ({', '.join(labels_dir)}):")
    print(f"   Levene Statistic = {stat_d:.4f}")
    print(f"   P-value          = {p_d:.4e}")
    
    if p_d < 0.05: 
        print("   -> RESULT: SIGNIFICANT. Changing direction AFFECTS precision.")
    else: 
        print("   -> RESULT: NOT Significant. Precision is consistent across directions.")

    # -------------------------------------------------------
    # Conclusion
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("COMPARISON OF EFFECTS:")
    
    # We compare the Test Statistics (larger F-stat usually implies a stronger effect size)
    # or P-values (smaller P-value implies higher significance).
    if stat_s > stat_d:
        print(f"Object SIZE (Stat={stat_s:.2f}) has a larger effect on variance than Pose (Stat={stat_d:.2f}).")
    else:
        print(f"Placing POSE (Stat={stat_d:.2f}) has a larger effect on variance than Size (Stat={stat_s:.2f}).")
    print("="*60)

if __name__ == "__main__":
    run()