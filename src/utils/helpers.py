import torch
from torch import nn
from pathlib import Path

def save_model(model: nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with .pt or .pth"

    model_save_path = target_dir_path / model_name
    torch.save(model.state_dict(), model_save_path)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions."""
    correct = (y_true == y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc

def print_train_time(start, end, device=None):
    """Prints the training time."""
    total_time = end - start
    device_info = f" on {device}" if device else ""
    print(f"Train time: {total_time:.3f} seconds{device_info}")
