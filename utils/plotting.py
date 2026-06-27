import matplotlib.pyplot as plt
import numpy as np

def apply_dark_theme(ax):
    """
    Applies a premium, publication-quality dark aesthetic to a matplotlib axis.
    """
    ax.set_facecolor('#121212')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#888888', labelsize=9)
    ax.grid(True, color='#222222', linestyle='--', linewidth=0.5)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('#ffffff')

def plot_scatter_steps(samples_dict, save_path, dataset_name=""):
    """
    Plots a 2x2 grid of scatter plots representing generated samples at steps [1, 5, 10, 20].
    
    Args:
        samples_dict (dict): Dictionary mapping steps (int) to samples (numpy array of shape (N, 2)).
        save_path (str): File path to save the generated image.
        dataset_name (str): Name of the dataset for plot headers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='#121212')
    steps_list = [1, 5, 10, 20]
    
    # Custom color palette matching the 8 gaussian clusters (HSL-like mapped to scatter points)
    for i, step in enumerate(steps_list):
        ax = axes[i // 2, i % 2]
        apply_dark_theme(ax)
        
        samples = samples_dict[step]
        # Calculate radius from origin for coloring scatter points beautifully
        radii = np.linalg.norm(samples, axis=1)
        
        # Draw scatter points with alpha transparency
        scatter = ax.scatter(
            samples[:, 0], samples[:, 1], 
            c=radii, cmap='plasma', 
            s=4, alpha=0.6, edgecolors='none'
        )
        
        ax.set_title(f"ODE Solver Steps: {step}", fontsize=12, fontweight='bold')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        
    title_text = "Generated Samples vs. Integration Steps"
    if dataset_name:
        title_text += f"\n[Dataset: {dataset_name.upper()}]"
        
    plt.suptitle(title_text, color='#ffffff', fontsize=15, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Plotting] Scatter plot saved to: {save_path}")

def plot_trajectories(trajectory, save_path, dataset_name="", num_paths=100):
    """
    Plots the integration trajectories of samples from noise (t=0) to data (t=1).
    
    Args:
        trajectory (numpy array): Trajectory array of shape (steps + 1, N, 2).
        save_path (str): File path to save the generated image.
        dataset_name (str): Name of the dataset for plot titles.
        num_paths (int): Number of trajectory paths to draw (default 100).
    """
    T, N, _ = trajectory.shape
    num_paths = min(num_paths, N)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    apply_dark_theme(ax)
    
    # Plot a subset of trajectories
    for i in range(num_paths):
        path = trajectory[:, i, :] # Shape (T, 2)
        
        # Map trajectory progress to a color gradient (cyan at t=0 to hot pink/orange at t=1)
        for t in range(T - 1):
            alpha = 0.15 + 0.35 * (t / T) # fade in paths as they converge
            color_ratio = t / (T - 1)
            # Interpolate color between cyan (#00f0ff) and hot pink (#ff007f)
            r = color_ratio
            g = 0.8 * (1 - color_ratio)
            b = 1.0
            ax.plot(
                path[t:t+2, 0], path[t:t+2, 1], 
                color=(r, g, b), alpha=alpha, linewidth=1.2
            )
            
    # Mark start points (noise at t=0) in cyan
    ax.scatter(trajectory[0, :num_paths, 0], trajectory[0, :num_paths, 1], color='#00f0ff', s=15, zorder=3, label='Noise (t=0)')
    
    # Mark end points (data at t=1) in gold
    ax.scatter(trajectory[-1, :num_paths, 0], trajectory[-1, :num_paths, 1], color='#ffd700', s=15, zorder=3, label='Generated (t=1)')
    
    title_text = "Flow Trajectories (t=0 to t=1)"
    if dataset_name:
        title_text += f"\n[Dataset: {dataset_name.upper()}]"
        
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#ffffff', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Plotting] Trajectory plot saved to: {save_path}")

def plot_pairing_comparison(pairs, trajectory, save_path, dataset_name="", num_paths=100):
    """
    Plots a comparison of:
    1. Straight line couplings between training pairs (x_0, x_1).
    2. The model's actual integration trajectories starting from x_0.
    
    Args:
        pairs (numpy array): Training pairs of shape (N, 2, 2).
        trajectory (numpy array): Actual trajectories of shape (T, N, 2).
        save_path (str): File path to save the generated image.
        dataset_name (str): Name of the dataset for plot titles.
        num_paths (int): Number of paths to visualize.
    """
    N = pairs.shape[0]
    T = trajectory.shape[0]
    num_paths = min(num_paths, N)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor='#121212')
    
    # 1. Plot Straight Line Couplings
    ax1 = axes[0]
    apply_dark_theme(ax1)
    ax1.set_title("Training Coupling Pairs (Straight Lines)", fontsize=13, fontweight='bold')
    
    for i in range(num_paths):
        x_0 = pairs[i, 0]
        x_1 = pairs[i, 1]
        ax1.plot([x_0[0], x_1[0]], [x_0[1], x_1[1]], color='#00f0ff', alpha=0.25, linewidth=1.0)
        
    ax1.scatter(pairs[:num_paths, 0, 0], pairs[:num_paths, 0, 1], color='#00f0ff', s=15, zorder=3, label='Noise (t=0)')
    ax1.scatter(pairs[:num_paths, 1, 0], pairs[:num_paths, 1, 1], color='#ffd700', s=15, zorder=3, label='Target (t=1)')
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)
    ax1.set_aspect('equal')
    ax1.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#ffffff')
    
    # 2. Plot Actual Model Trajectories
    ax2 = axes[1]
    apply_dark_theme(ax2)
    ax2.set_title("Model Generated Trajectories (Curve Path)", fontsize=13, fontweight='bold')
    
    for i in range(num_paths):
        path = trajectory[:, i, :]
        for t in range(T - 1):
            alpha = 0.15 + 0.35 * (t / T)
            color_ratio = t / (T - 1)
            r = color_ratio
            g = 0.8 * (1 - color_ratio)
            b = 1.0
            ax2.plot(path[t:t+2, 0], path[t:t+2, 1], color=(r, g, b), alpha=alpha, linewidth=1.2)
            
    ax2.scatter(trajectory[0, :num_paths, 0], trajectory[0, :num_paths, 1], color='#00f0ff', s=15, zorder=3, label='Start (t=0)')
    ax2.scatter(trajectory[-1, :num_paths, 0], trajectory[-1, :num_paths, 1], color='#ffd700', s=15, zorder=3, label='Generated (t=1)')
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal')
    ax2.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#ffffff')
    
    suptitle_text = "Reflow Pairing Analysis: Couplings vs. Model Trajectories"
    if dataset_name:
        suptitle_text += f"\n[Dataset: {dataset_name.upper()}]"
        
    plt.suptitle(suptitle_text, color='#ffffff', fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Plotting] Pairing comparison plot saved to: {save_path}")
