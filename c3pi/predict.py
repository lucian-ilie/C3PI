import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import List

def predict(model: Module, dataloader: DataLoader, device: torch.device) -> Tensor:
    """
    Run inference on the given model and dataloader.

    Args:
        model (torch.nn.Module): Trained PPI model.
        dataloader (DataLoader): PyTorch dataloader for the prediction data.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        torch.Tensor: Concatenated prediction results.
    """
    model.eval()
    predictions: List[Tensor] = []
    with torch.no_grad():
        for input1, input2, _ in dataloader:
            input1 = input1.to(device)
            input2 = input2.to(device)
            outputs = model(input1, input2)
            predictions.append(outputs.cpu())
    return torch.cat(predictions, dim=0)
