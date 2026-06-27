import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path to allow absolute imports from any subfolder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import hyperparameters as hp
from utils.hardware import get_device, prepare_model
from models.velocity_net import VelocityNet

class ReflowDataset(Dataset):
    def __init__(self, file_path):
        # Load array of shape (N, 2, 2)
        # pairs[i, 0] is x_0, pairs[i, 1] is x_1_flow
        self.pairs = np.load(file_path).astype(np.float32)
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        x_0 = torch.tensor(self.pairs[idx, 0], dtype=torch.float32)
        x_1_flow = torch.tensor(self.pairs[idx, 1], dtype=torch.float32)
        return x_0, x_1_flow

def train_reflow():
    print("=== Phase 2: Training Reflow Model ===")
    
    # 1. Setup hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # 2. Setup dataloader
    pairs_path = os.path.join(hp.DATA_DIR, "reflow_pairs.npy")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Reflow pairs data not found at: {pairs_path}. Please run generate_pairs.py first.")
        
    dataset = ReflowDataset(pairs_path)
    dataloader = DataLoader(dataset, batch_size=hp.BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} paired samples.")
    
    # 3. Initialize model
    model = VelocityNet(hidden_dim=hp.HIDDEN_DIM, time_embed_dim=hp.TIME_EMBED_DIM)
    model = prepare_model(model, device)
    
    # 4. Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    
    # 5. Training loop
    model.train()
    for epoch in range(hp.EPOCHS_REFLOW):
        epoch_loss = 0.0
        for batch_idx, (x_0, x_1_flow) in enumerate(dataloader):
            x_0 = x_0.to(device)
            x_1_flow = x_1_flow.to(device)
            
            optimizer.zero_grad()
            
            # Sample t ~ U(0, 1)
            batch_size = x_0.size(0)
            t = torch.rand(batch_size, device=device)
            t_expanded = t.view(-1, 1)
            
            # Interpolate: x_t = t * x_1_flow + (1 - t) * x_0
            x_t = t_expanded * x_1_flow + (1.0 - t_expanded) * x_0
            
            # Target velocity for Reflow path: x_1_flow - x_0
            target_v = x_1_flow - x_0
            
            # Model prediction: v_theta(x_t, t)
            pred_v = model(x_t, t)
            
            # Reflow Loss: MSE
            loss = torch.mean((pred_v - target_v) ** 2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % hp.PRINT_INTERVAL == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{hp.EPOCHS_REFLOW:03d} | Average Loss: {avg_loss:.6f}")
            
    # 6. Save checkpoint
    checkpoint_path = os.path.join(hp.CHECKPOINT_DIR, "reflow.pt")
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    torch.save({
        'epoch': hp.EPOCHS_REFLOW,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'hidden_dim': hp.HIDDEN_DIM,
            'time_embed_dim': hp.TIME_EMBED_DIM,
            'radius': hp.RADIUS,
            'std': hp.STD
        }
    }, checkpoint_path)
    
    print(f"Saved Reflow checkpoint to: {checkpoint_path}")
    print("=== Phase 2 Training Completed Successfully ===")

if __name__ == "__main__":
    train_reflow()
