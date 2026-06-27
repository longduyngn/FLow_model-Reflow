import torch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def prepare_model(model, device):
    model = model.to(device)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[Hardware] CUDA is available. Found {gpu_count} GPU(s).")
        if gpu_count >= 2:
            print(f"[Hardware] Wrapping model in torch.nn.DataParallel across {gpu_count} GPUs.")
            model = torch.nn.DataParallel(model)
    else:
        print("[Hardware] Running on CPU.")
    return model

def to_device(data, device):
    """
    Recursively sends PyTorch Tensors or collections of Tensors to the target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data
