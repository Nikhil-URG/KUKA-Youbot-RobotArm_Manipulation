import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

def run():
    try:
        df = pd.read_csv("combined_transformed_data.csv")
    except FileNotFoundError:
        print("Error: csv file not found.")
        return

    # Create output directory
    os.makedirs("plots", exist_ok=True)

    print("\n" + "="*60)
    print(" STEP 2: PCA ANALYSIS (X, Y) - OPTITRACK DATA")
    print("="*60)
    
    for direction in df['Direction'].unique():
        subset = df[df['Direction'] == direction]
        
        # USE OPTITRACK DATA for the actual spread analysis
        # Dropping NaNs is crucial
        data = subset[['x_optitrack', 'y_optitrack']].dropna().values
        
        if len(data) < 2: 
            print(f"Skipping {direction}: Not enough data.")
            continue
        
        # 1. PCA Calculation
        mean = np.mean(data, axis=0)
        centered = data - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        
        # Sort by eigenvalue size (descending)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # Calculate Angle of the principal component
        angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
        
        print(f"Direction: {direction}")
        print(f"  Eigenvalues: {eigvals}")
        print(f"  Principal Vector: {eigvecs[:,0]}")

        # 2. Plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter points
        ax.scatter(data[:,0], data[:,1], alpha=0.6, s=20, label='End Poses (OptiTrack)', color='blue')
        
        # Draw 2-Sigma Ellipse (approx 95% Confidence Interval)
        # Width and Height are 2 * n_sigma * sqrt(eigenvalue)
        n_sigma = 2
        w = 2 * n_sigma * np.sqrt(eigvals[0])
        h = 2 * n_sigma * np.sqrt(eigvals[1])
        
        ell = Ellipse(xy=mean, width=w, height=h, angle=angle, 
                      edgecolor='red', facecolor='none', lw=2, linestyle='--', label='95% Conf. Ellipse')
        ax.add_patch(ell)
        
        # Plot Mean
        ax.scatter(mean[0], mean[1], c='red', marker='x', s=100, label='Mean Position')
        
        # Draw Principal Axes
        # Multiply eigenvector by 3*sigma for visualization length
        v1 = eigvecs[:,0] * 3 * np.sqrt(eigvals[0])
        v2 = eigvecs[:,1] * 3 * np.sqrt(eigvals[1])
        
        ax.arrow(mean[0], mean[1], v1[0], v1[1], color='green', width=0.02, head_width=0.1, label='Principal Axis 1')
        ax.arrow(mean[0], mean[1], v2[0], v2[1], color='orange', width=0.02, head_width=0.1, label='Principal Axis 2')

        # 3. Smart Axis Scaling (Zoom in)
        # We find the min/max of the data and add a small padding
        x_min, x_max = data[:,0].min(), data[:,0].max()
        y_min, y_max = data[:,1].min(), data[:,1].max()
        
        # Determine the larger range to keep aspect ratio square
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        padding = max_range * 0.2  # 20% padding
        
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        
        ax.set_xlim(mid_x - max_range/2 - padding, mid_x + max_range/2 + padding)
        ax.set_ylim(mid_y - max_range/2 - padding, mid_y + max_range/2 + padding)
        
        ax.set_title(f'PCA Analysis: {direction} Motion\n(OptiTrack Data)', fontsize=14)
        ax.set_xlabel('X Position [cm]', fontsize=12)
        ax.set_ylabel('Y Position [cm]', fontsize=12)
        ax.axis('equal') # IMPORTANT: Keeps circles circular
        ax.legend()
        plt.grid(True, alpha=0.3)
        
        # Save Plot
        filename = f"plots/pca_{direction}.png"
        plt.savefig(filename, dpi=300)
        print(f"  -> Plot saved to {filename}")
        plt.close()

if __name__ == "__main__":
    run()