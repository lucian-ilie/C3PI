import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import logging
from torch.utils.data import DataLoader
from typing import Tuple



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = 'best_model.pth',
    log_dir: str = 'runs/default'
) -> None:
    """
    Train a binary classification model using two-input data.

    Args:
        model (nn.Module): The PyTorch model to train. It must accept two inputs.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        save_path (str, optional): File path to save the best model. Defaults to 'best_model.pth'.
        log_dir (str, optional): Directory for TensorBoard logs. Defaults to 'runs/default'.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir)
    logging.info(f"Using device: {device}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for input1, input2, labels in train_loader:
            input1, input2 = input1.to(device), input2.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input1.size(0)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        writer.add_scalar('Loss/Train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        log_msg = (f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_epoch_loss:.4f} - "
                   f"Val Loss: {val_loss:.4f} - "
                   f"Val Acc: {val_acc:.4f} - "
                   f"Time: {(time.time()-start_time)/60:.2f} min")
        logging.info(log_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f"Model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

    writer.close()


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The PyTorch model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on (CPU or CUDA).

    Returns:
        Tuple[float, float]: A tuple containing validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for input1, input2, labels in val_loader:
            input1, input2 = input1.to(device), input2.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * input1.size(0)

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return val_loss / len(val_loader.dataset), correct / total

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataloader import PPIDatasetFromFiles, protein_collate_fn
    from c3pi import FullPPIModel

    def test_single_batch():
        # Load a small subset of data for quick test
        dataset = PPIDatasetFromFiles(
            '/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/pairs/human_train_balanced.tsv',
            '/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/embd/human/')
        loader = DataLoader(dataset, batch_size=2, collate_fn=protein_collate_fn)

        model = FullPPIModel(input_features_dim=1024)
        model.eval()
        for input1, input2, labels in loader:
            output = model(input1, input2)
            assert output.shape == (16, 1), f"Expected output shape (16, 1), got {output.shape}"
            print("Forward pass works. Output:", output.detach().cpu().numpy())
            break  # Only test the first batch

    test_single_batch()
