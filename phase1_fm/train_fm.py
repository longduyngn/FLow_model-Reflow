import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.optim as optim

# Add project root to sys.path to allow absolute imports from any subfolder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import hyperparameters as hp
from utils.hardware import get_device, prepare_model
from data.gaussian_8mode import get_dataloader
from models.velocity_net import VelocityNet
from core.fm_loss import compute_fm_loss

def train():
    print("=== Phase 1: Training Base Flow Matching Model ===")
    
    # 1.  hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # 2.  dataloader
    dataloader = get_dataloader(
        dataset_type=hp.DATASET_TYPE,
        batch_size=hp.BATCH_SIZE, 
        num_samples=hp.NUM_SAMPLES_DATA, 
        radius=hp.RADIUS, 
        std=hp.STD, 
        shuffle=True
    )
    
    # 3. Initialize model
    model = VelocityNet(hidden_dim=hp.HIDDEN_DIM, time_embed_dim=hp.TIME_EMBED_DIM)
    model = prepare_model(model, device) # FOR  kaggle wrapping  DataParallel
    
    # 4. optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    
    # 5. Training loop
    model.train()
    for epoch in range(hp.EPOCHS_FM):
        epoch_loss = 0.0
        for batch_idx, x_1 in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Loss computation & backprop
            loss = compute_fm_loss(model, x_1, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{hp.EPOCHS_FM:03d} | Average Loss: {avg_loss:.6f}")
            
    # 6. Save checkpoint (stripping DataParallel wrapper if present)
    os.makedirs(hp.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(hp.CHECKPOINT_DIR, "fm_base.pt")
    
    # Un-wrap model from DataParallel
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    torch.save({
        'epoch': hp.EPOCHS_FM,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'hidden_dim': hp.HIDDEN_DIM,
            'time_embed_dim': hp.TIME_EMBED_DIM,
            'radius': hp.RADIUS,
            'std': hp.STD
        }
    }, checkpoint_path)
    
    print(f"Saved base FM checkpoint to: {checkpoint_path}")
    print("=== Phase 1 Training Completed Successfully ===")

if __name__ == "__main__":
    train()
