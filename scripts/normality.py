import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run():
    # 1. Load the data
    try:
        df = pd.read_csv("combined_transformed_data.csv")
    except FileNotFoundError:
        print("Error: combined_transformed_data.csv not found. Run common_data_loader.py first.")
        return

    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)

    print("\n" + "="*60)
    print(" STEP 1: GAUSSIAN CHECK (Chi-Square) on OPTITRACK DATA")
    print("="*60)
    
    # We analyze the ACTUAL position (OptiTrack) to see the spread
    for direction in df['Direction'].unique():
        subset = df[df['Direction'] == direction]
        
        # CHANGE: Analyze 'x_optitrack' instead of 'x_youbot_transformed'
        data = subset['x_optitrack'].dropna()
        
        if len(data) < 5:
            print(f"Skipping {direction}: Not enough data points ({len(data)}).")
            continue

        # --- Chi-Square Test ---
        # Dynamic bin count
        n_bins = max(5, int(np.sqrt(len(data))))
        observed, bin_edges = np.histogram(data, bins=n_bins)
        
        # Calculate Expected Frequencies
        mu, std = stats.norm.fit(data)
        
        # Handle Zero Variance Case
        if std == 0:
            print(f"Direction: {direction:<10} | Variance is 0. Cannot fit Gaussian.")
            continue

        cdf = stats.norm.cdf(bin_edges, loc=mu, scale=std)
        expected_probs = np.diff(cdf)
        expected = expected_probs * len(data)
        expected = expected * (observed.sum() / expected.sum()) # Normalize
        
        # Perform Test
        stat, p = stats.chisquare(observed, expected, ddof=2)
        
        res = "Reject H0 (Not Gaussian)" if p < 0.05 else "Fail to Reject (Gaussian)"
        print(f"Direction: {direction:<10} | P-value: {p:.4e} -> {res}")
        
        # --- Plotting ---
        plt.figure(figsize=(8, 6))
        
        # Plot Histogram
        sns.histplot(data, kde=False, stat='density', label='Observed Data', color='#87CEEB', edgecolor='black', bins=n_bins)
        
        # Plot Gaussian Curve
        xmin, xmax = plt.xlim()
        x_plot = np.linspace(xmin, xmax, 100)
        p_plot = stats.norm.pdf(x_plot, mu, std)
        plt.plot(x_plot, p_plot, 'r-', linewidth=3, label=f'Gaussian Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
        
        plt.title(f'Normality Check: {direction} Motion (OptiTrack)\n(Chi-Sq p={p:.4e} -> {res})', fontsize=14)
        plt.xlabel('OptiTrack X Position (cm)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save Plot
        filename = f"plots/normality_{direction}.png"
        plt.savefig(filename, dpi=300)
        print(f"  -> Plot saved to {filename}")
        plt.close() 

if __name__ == "__main__":
    run()