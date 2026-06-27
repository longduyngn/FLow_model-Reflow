import torch

@torch.no_grad()
def solve_ode(model, x_start, steps=20, return_trajectory=False, direction="forward"):
    """
    Solves the ODE dx = v(x, t) dt using Euler's method.
    
    Args:
        model (nn.Module): The velocity field neural network.
        x_start (Tensor): Starting points. If direction is 'forward', points at t=0.
                          If direction is 'backward', points at t=1.
        steps (int): Number of integration steps.
        return_trajectory (bool): If True, returns the full integration path.
        direction (str): 'forward' to solve from t=0 to t=1, or 'backward' from t=1 to t=0.
        
    Returns:
        Tensor: Final points at the end of integration (t=1 for forward, t=0 for backward).
        Tensor (optional): Trajectory tensor of shape (steps + 1, num_samples, 2).
    """
    device = x_start.device
    x = x_start.clone()
    
    trajectory = [x.clone()] if return_trajectory else None
    
    if direction == "forward":
        dt = 1.0 / steps
        for step in range(steps):
            t_val = step / steps
            t = torch.full((x.size(0),), t_val, device=device, dtype=x.dtype)
            v = model(x, t)
            x = x + v * dt
            if return_trajectory:
                trajectory.append(x.clone())
                
    elif direction == "backward":
        # Integrating backwards from t=1 to t=0
        dt = -1.0 / steps
        for step in reversed(range(1, steps + 1)):
            t_val = step / steps
            t = torch.full((x.size(0),), t_val, device=device, dtype=x.dtype)
            v = model(x, t)
            x = x + v * dt
            if return_trajectory:
                trajectory.append(x.clone())
    else:
        raise ValueError(f"Unknown direction: {direction}")
            
    if return_trajectory:
        return x, torch.stack(trajectory, dim=0)
    else:
        return x
