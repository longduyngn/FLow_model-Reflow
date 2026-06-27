import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import numpy as np

# Add project root to sys.path to allow absolute imports from any subfolder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import hyperparameters as hp
from utils.hardware import get_device, prepare_model
from models.velocity_net import VelocityNet
from core.ode_solver import solve_ode

def generate_pairs():
    print("=== Generating Pairs for Reflow ===")
    
    # 1. Setup hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # 2. Load checkpoint
    checkpoint_path = os.path.join(hp.CHECKPOINT_DIR, "fm_base.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base FM checkpoint not found at: {checkpoint_path}. Please train Phase 1 first.")
        
    print(f"Loading base FM checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Read saved hyperparameters to instantiate model correctly
    saved_hp = checkpoint.get('hyperparameters', {})
    hidden_dim = saved_hp.get('hidden_dim', hp.HIDDEN_DIM)
    time_embed_dim = saved_hp.get('time_embed_dim', hp.TIME_EMBED_DIM)
    
    # 3. Instantiate and load model
    model = VelocityNet(hidden_dim=hidden_dim, time_embed_dim=time_embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = prepare_model(model, device)
    model.eval()
    
    # 4. Set seed and read creation mode
    torch.manual_seed(42)
    np.random.seed(42)
    
    pair_mode = getattr(hp, 'REFLOW_PAIR_MODE', 'forward')
    print(f"Reflow pair creation mode: {pair_mode}")
    
    if pair_mode == 'backward':
        # Sample real data points x_1 from distribution
        print(f"Sampling {hp.NUM_SAMPLES_REFLOW} real data points x_1 ~ q(x_1) using {hp.DATASET_TYPE}")
        from data.gaussian_8mode import ToyDataset
        dataset = ToyDataset(dataset_type=hp.DATASET_TYPE, num_samples=hp.NUM_SAMPLES_REFLOW, radius=hp.RADIUS, std=hp.STD)
        x_1 = torch.tensor(dataset.data, dtype=torch.float32, device=device)
        
        # Integrate real data backward to t=0
        print(f"Integrating real data backward to noise using base FM model (steps={hp.ODE_STEPS})...")
        batch_size = 500
        x_0_flow_list = []
        
        for i in range(0, hp.NUM_SAMPLES_REFLOW, batch_size):
            x_1_batch = x_1[i:i+batch_size]
            x_0_batch = solve_ode(model, x_1_batch, steps=hp.ODE_STEPS, return_trajectory=False, direction="backward")
            x_0_flow_list.append(x_0_batch.cpu())
            
        x_0_flow = torch.cat(x_0_flow_list, dim=0)
        x_1_cpu = x_1.cpu()
        
        # pair_arr[i, 0] = x_0_flow (noise), pair_arr[i, 1] = x_1 (real data)
        pair_arr = np.stack([x_0_flow.numpy(), x_1_cpu.numpy()], axis=1)
    else:
        # Generate noise x_0
        print(f"Sampling {hp.NUM_SAMPLES_REFLOW} noise vectors x_0 ~ N(0, I)")
        x_0 = torch.randn(hp.NUM_SAMPLES_REFLOW, 2, device=device)
        
        # Integrate noise forward to t=1
        print(f"Integrating noise forward to target using base FM model (steps={hp.ODE_STEPS})...")
        batch_size = 500
        x_1_flow_list = []
        
        for i in range(0, hp.NUM_SAMPLES_REFLOW, batch_size):
            x_0_batch = x_0[i:i+batch_size]
            x_1_batch = solve_ode(model, x_0_batch, steps=hp.ODE_STEPS, return_trajectory=False, direction="forward")
            x_1_flow_list.append(x_1_batch.cpu())
            
        x_1_flow = torch.cat(x_1_flow_list, dim=0)
        x_0_cpu = x_0.cpu()
        
        # pair_arr[i, 0] = x_0 (noise), pair_arr[i, 1] = x_1_flow (generated data)
        pair_arr = np.stack([x_0_cpu.numpy(), x_1_flow.numpy()], axis=1)
        
    os.makedirs(hp.DATA_DIR, exist_ok=True)
    save_path = os.path.join(hp.DATA_DIR, "reflow_pairs.npy")
    np.save(save_path, pair_arr)
    
    print(f"Successfully generated and saved {len(pair_arr)} pairs to: {save_path}")
    print(f"Data shape: {pair_arr.shape}")
    print("=== Pair Generation Completed ===")

if __name__ == "__main__":
    generate_pairs()
