import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import sys

# Set python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.gaussian_8mode import get_dataloader, Gaussian8ModeDataset
from models.layers import SinusoidalEmbedding
from models.velocity_net import VelocityNet
from core.fm_loss import compute_fm_loss
from core.ode_solver import solve_ode

def test_dataloader():
    print("Testing Dataloader for all dataset types...")
    dataset_types = ["gaussian_8mode", "two_moons", "swiss_roll", "checkerboard"]
    for dtype in dataset_types:
        dataloader = get_dataloader(dataset_type=dtype, batch_size=32, num_samples=128)
        batch = next(iter(dataloader))
        assert batch.shape == (32, 2), f"Expected shape (32, 2) for {dtype}, got {batch.shape}"
        print(f"  {dtype}: OK")
    print("Dataloader: OK")

def test_sinusoidal_embedding():
    print("Testing Sinusoidal Embedding...")
    embed = SinusoidalEmbedding(dim=64)
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = embed(t)
    assert emb.shape == (3, 64), f"Expected shape (3, 64), got {emb.shape}"
    print("Sinusoidal Embedding: OK")

def test_velocity_net():
    print("Testing VelocityNet...")
    model = VelocityNet(hidden_dim=128, time_embed_dim=32)
    x = torch.randn(16, 2)
    t = torch.rand(16)
    v = model(x, t)
    assert v.shape == (16, 2), f"Expected shape (16, 2), got {v.shape}"
    print("VelocityNet: OK")

def test_fm_loss():
    print("Testing FM Loss...")
    model = VelocityNet(hidden_dim=128, time_embed_dim=32)
    x_1 = torch.randn(8, 2)
    device = torch.device("cpu")
    loss = compute_fm_loss(model, x_1, device)
    assert loss.dim() == 0, f"Expected scalar loss, got dim {loss.dim()}"
    print("FM Loss: OK")

def test_ode_solver():
    print("Testing ODE Solver...")
    model = VelocityNet(hidden_dim=128, time_embed_dim=32)
    x_0 = torch.randn(10, 2)
    
    # Check regular solve (forward)
    x_1 = solve_ode(model, x_0, steps=20, return_trajectory=False, direction="forward")
    assert x_1.shape == (10, 2), f"Expected shape (10, 2), got {x_1.shape}"
    
    # Check backward solve
    x_0_back = solve_ode(model, x_1, steps=20, return_trajectory=False, direction="backward")
    assert x_0_back.shape == (10, 2), f"Expected shape (10, 2), got {x_0_back.shape}"
    
    # Check trajectory solve (forward)
    x_1_traj, traj = solve_ode(model, x_0, steps=20, return_trajectory=True, direction="forward")
    assert x_1_traj.shape == (10, 2)
    assert traj.shape == (21, 10, 2), f"Expected shape (21, 10, 2), got {traj.shape}"
    print("ODE Solver: OK")

if __name__ == "__main__":
    print("=== Running Sanity Tests ===")
    test_dataloader()
    test_sinusoidal_embedding()
    test_velocity_net()
    test_fm_loss()
    test_ode_solver()
    print("=== All Sanity Tests Passed! ===")
