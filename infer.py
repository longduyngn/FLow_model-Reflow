import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import hyperparameters as hp
from utils.hardware import get_device, prepare_model
from models.velocity_net import VelocityNet
from core.ode_solver import solve_ode
from utils.plotting import plot_scatter_steps, plot_trajectories, plot_pairing_comparison

def compute_straightness(trajectory_np):
    """
    Computes the path straightness metric: straight_line_distance / arc_length.
    Perfect straight path = 1.0. Curved paths < 1.0.
    trajectory_np shape: (T, N, 2)
    """
    T, N, _ = trajectory_np.shape
    # Straight-line distance between t=0 and t=1
    straight_dist = np.linalg.norm(trajectory_np[-1] - trajectory_np[0], axis=1) # (N,)
    
    # Calculate consecutive differences
    diffs = np.diff(trajectory_np, axis=0) # (T-1, N, 2)
    step_lens = np.linalg.norm(diffs, axis=2) # (T-1, N)
    arc_len = np.sum(step_lens, axis=0) # (N,)
    
    # Straightness ratio
    straightness = np.zeros_like(straight_dist)
    valid = arc_len > 1e-6
    straightness[valid] = straight_dist[valid] / arc_len[valid]
    return np.mean(straightness)

def run_inference():
    print("=== Flow Matching & Reflow Inference ===")
    
    # 1. Setup hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # 2. Checkpaths
    fm_checkpoint_path = os.path.join(hp.CHECKPOINT_DIR, "fm_base.pt")
    reflow_checkpoint_path = os.path.join(hp.CHECKPOINT_DIR, "reflow.pt")
    pairs_path = os.path.join(hp.DATA_DIR, "reflow_pairs.npy")
    
    if not os.path.exists(fm_checkpoint_path):
        print(f"[Warning] Base FM checkpoint not found at {fm_checkpoint_path}.")
    if not os.path.exists(reflow_checkpoint_path):
        print(f"[Warning] Reflow checkpoint not found at {reflow_checkpoint_path}.")
        
    # Seed for evaluation
    torch.manual_seed(999)
    np.random.seed(999)
    
    # Number of samples for visualization
    num_samples = 5000
    x_0 = torch.randn(num_samples, 2, device=device)
    
    # Load reflow pairs for comparison if they exist
    pairs = None
    if os.path.exists(pairs_path):
        print(f"Loading saved Reflow pairs from: {pairs_path}")
        pairs = np.load(pairs_path) # (N, 2, 2)
        
    # ==========================================
    # Run Base Flow Matching Model Inference
    # ==========================================
    fm_straightness = None
    if os.path.exists(fm_checkpoint_path):
        print("\n--- Running Base Flow Matching Inference ---")
        checkpoint = torch.load(fm_checkpoint_path, map_location=device)
        model = VelocityNet(hidden_dim=hp.HIDDEN_DIM, time_embed_dim=hp.TIME_EMBED_DIM)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = prepare_model(model, device)
        model.eval()
        
        # Test steps: 1, 5, 10, 20
        fm_samples_dict = {}
        for steps in [1, 5, 10, 20]:
            print(f"Sampling with ODE steps = {steps}...")
            samples = solve_ode(model, x_0, steps=steps, return_trajectory=False)
            fm_samples_dict[steps] = samples.cpu().numpy()
            
        # Draw scatter plots
        os.makedirs(hp.OUTPUT_DIR, exist_ok=True)
        plot_scatter_steps(fm_samples_dict, os.path.join(hp.OUTPUT_DIR, "fm_scatter.png"), hp.DATASET_TYPE)
        
        # Draw trajectories for steps = 20
        _, trajectory = solve_ode(model, x_0, steps=hp.ODE_STEPS, return_trajectory=True)
        trajectory_np = trajectory.cpu().numpy()
        plot_trajectories(trajectory_np, os.path.join(hp.OUTPUT_DIR, "fm_trajectories.png"), hp.DATASET_TYPE, num_paths=120)
        
        fm_straightness = compute_straightness(trajectory_np)
        print(f"[Metrics] Base FM Mean Path Straightness: {fm_straightness:.5f} (1.0 is perfectly straight)")
        
        # Plot pairing comparison
        if pairs is not None:
            print("Generating pairing comparison plot for Base FM...")
            x_0_test = torch.tensor(pairs[:500, 0], dtype=torch.float32, device=device)
            _, test_traj = solve_ode(model, x_0_test, steps=hp.ODE_STEPS, return_trajectory=True)
            plot_pairing_comparison(
                pairs[:500], 
                test_traj.cpu().numpy(), 
                os.path.join(hp.OUTPUT_DIR, "pairing_comparison_fm.png"),
                hp.DATASET_TYPE,
                num_paths=100
            )
    else:
        fm_straightness = None
        
    # ==========================================
    # Run Reflow Model Inference
    # ==========================================
    reflow_straightness = None
    if os.path.exists(reflow_checkpoint_path):
        print("\n--- Running Reflow Inference ---")
        checkpoint = torch.load(reflow_checkpoint_path, map_location=device)
        model = VelocityNet(hidden_dim=hp.HIDDEN_DIM, time_embed_dim=hp.TIME_EMBED_DIM)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = prepare_model(model, device)
        model.eval()
        
        # Test steps: 1, 5, 10, 20
        reflow_samples_dict = {}
        for steps in [1, 5, 10, 20]:
            print(f"Sampling with ODE steps = {steps}...")
            samples = solve_ode(model, x_0, steps=steps, return_trajectory=False)
            reflow_samples_dict[steps] = samples.cpu().numpy()
            
        # Draw scatter plots
        plot_scatter_steps(reflow_samples_dict, os.path.join(hp.OUTPUT_DIR, "reflow_scatter.png"), hp.DATASET_TYPE)
        
        # Draw trajectories for steps = 20
        _, trajectory = solve_ode(model, x_0, steps=hp.ODE_STEPS, return_trajectory=True)
        trajectory_np = trajectory.cpu().numpy()
        plot_trajectories(trajectory_np, os.path.join(hp.OUTPUT_DIR, "reflow_trajectories.png"), hp.DATASET_TYPE, num_paths=120)
        
        reflow_straightness = compute_straightness(trajectory_np)
        print(f"[Metrics] Reflow Mean Path Straightness: {reflow_straightness:.5f} (1.0 is perfectly straight)")
        
        # Plot pairing comparison
        if pairs is not None:
            print("Generating pairing comparison plot for Reflow...")
            x_0_test = torch.tensor(pairs[:500, 0], dtype=torch.float32, device=device)
            _, test_traj = solve_ode(model, x_0_test, steps=hp.ODE_STEPS, return_trajectory=True)
            plot_pairing_comparison(
                pairs[:500], 
                test_traj.cpu().numpy(), 
                os.path.join(hp.OUTPUT_DIR, "pairing_comparison_reflow.png"),
                hp.DATASET_TYPE,
                num_paths=100
            )
    else:
        reflow_straightness = None
        
    # ==========================================
    # Comparison Analysis
    # ==========================================
    if fm_straightness is not None and reflow_straightness is not None:
        print("\n=======================================================")
        print("                 COMPARATIVE ANALYSIS                  ")
        print("=======================================================")
        print(f"Base Flow Matching Path Straightness: {fm_straightness:.5f}")
        print(f"Reflow (Rectified Flow) Straightness: {reflow_straightness:.5f}")
        improvement = (reflow_straightness - fm_straightness) / (1.0 - fm_straightness) * 100 if fm_straightness < 1.0 else 0
        print(f"Straightness Improvement: {improvement:.2f}% (relative to upper bound 1.0)")
        print("\nObservations:")
        print("1. The Base Flow Matching model integrates over curved paths because the noise x_0")
        print("   and data x_1 are not pre-coupled. The ODE solver paths can intersect and bend.")
        print("2. The Reflow model integrates over significantly straighter paths.")
        print("   This is because the training targets are deterministic, pre-coupled pairs (x_0, x_1_flow)")
        print("   generated by flowing noise through the base model.")
        print("3. Straighter paths allow high-fidelity generation in fewer integration steps")
        print("   (e.g., compare the quality at steps=5 and steps=10 between both models).")
        print("=======================================================")

if __name__ == "__main__":
    run_inference()
