import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def make_gaussian_8mode(n_samples=100000, radius=5.0, std=0.5):
    """Generates an 8-mode Gaussian mixture arranged in a circle."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    cluster_indices = np.random.randint(0, 8, size=n_samples)
    centers_sampled = centers[cluster_indices]
    noise = np.random.randn(n_samples, 2) * std
    return (centers_sampled + noise).astype(np.float32)

def make_moons(n_samples=100000, noise=0.1):
    """Generates two intertwined half circles (moons)."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    theta_out = np.linspace(0, np.pi, n_samples_out)
    theta_in = np.linspace(0, np.pi, n_samples_in)
    
    outer_x = np.cos(theta_out)
    outer_y = np.sin(theta_out)
    inner_x = 1.0 - np.cos(theta_in)
    inner_y = 0.5 - np.sin(theta_in)
    
    x = np.concatenate([outer_x, inner_x])
    y = np.concatenate([outer_y, inner_y])
    X = np.stack([x, y], axis=1)
    
    # Center and scale to range [-5, 5]
    X = X * 4.0 - np.array([2.0, 1.0])
    X += np.random.randn(*X.shape) * noise
    return X.astype(np.float32)

def make_swiss_roll(n_samples=100000, noise=0.1):
    """Generates a Swiss roll spiral distribution."""
    t = 1.5 * np.pi * (1.0 + 2.0 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    X = np.stack([x, y], axis=1)
    
    # Scale down to fit standard visualization coordinates
    X = X * 0.4
    X += np.random.randn(*X.shape) * noise
    return X.astype(np.float32)

def make_checkerboard(n_samples=100000, noise=0.05):
    """Generates a 4x4 checkerboard distribution on [-4, 4] x [-4, 4]."""
    x = np.random.uniform(-4, 4, size=n_samples)
    y = np.random.uniform(-4, 4, size=n_samples)
    
    grid_x = np.floor((x + 4) / 2.0).astype(int)
    grid_y = np.floor((y + 4) / 2.0).astype(int)
    
    # Check parity and project odd grid cells to adjacent even cells
    mask = (grid_x + grid_y) % 2 == 1
    x[mask] = x[mask] + 2.0
    overflow = x > 4.0
    x[overflow] = x[overflow] - 8.0
    
    X = np.stack([x, y], axis=1)
    X += np.random.randn(*X.shape) * noise
    return X.astype(np.float32)

class ToyDataset(Dataset):
    def __init__(self, dataset_type="gaussian_8mode", num_samples=100000, radius=5.0, std=0.5):
        """
        General 2D toy dataset builder.
        """
        self.num_samples = num_samples
        self.dataset_type = dataset_type
        
        if dataset_type == "gaussian_8mode":
            self.data = make_gaussian_8mode(num_samples, radius, std)
        elif dataset_type == "two_moons":
            # Scale std to appropriate noise ranges
            noise = std * 0.3 if std is not None else 0.15
            self.data = make_moons(num_samples, noise)
        elif dataset_type == "swiss_roll":
            noise = std * 0.2 if std is not None else 0.1
            self.data = make_swiss_roll(num_samples, noise)
        elif dataset_type == "checkerboard":
            noise = std * 0.15 if std is not None else 0.05
            self.data = make_checkerboard(num_samples, noise)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

class Gaussian8ModeDataset(ToyDataset):
    """
    Subclass of ToyDataset mapping specifically to 8-mode Gaussian.
    Maintains compatibility with pre-existing tests.
    """
    def __init__(self, num_samples=100000, radius=5.0, std=0.5):
        super().__init__(dataset_type="gaussian_8mode", num_samples=num_samples, radius=radius, std=std)

def get_dataloader(dataset_type="gaussian_8mode", batch_size=256, num_samples=100000, radius=5.0, std=0.5, shuffle=True):
    """
    Returns a PyTorch Dataloader for the specified 2D distribution.
    """
    dataset = ToyDataset(dataset_type=dataset_type, num_samples=num_samples, radius=radius, std=std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader
