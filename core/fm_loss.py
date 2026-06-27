import torch

def compute_fm_loss(model, x_1, device):
    """
    Computes Independent Target Flow Matching Loss.
    x_1 is the target data batch of shape (batch_size, 2).
    """
    batch_size = x_1.size(0)
    
    # 1. Sample noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(x_1).to(device)
    x_1 = x_1.to(device)
    
    # 2. Sample time t ~ U(0, 1)
    t = torch.rand(batch_size, device=device)
    
    # 3. Compute linear interpolation path: x_t = t * x_1 + (1 - t) * x_0
    t_expanded = t.view(-1, 1)
    x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
    
    # 4. Target velocity: d/dt(x_t) = x_1 - x_0
    target_v = x_1 - x_0
    
    # 5. Predict velocity field
    pred_v = model(x_t, t)
    
    # 6. Compute MSE Loss
    loss = torch.mean((pred_v - target_v) ** 2)
    return loss
