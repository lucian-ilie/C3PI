import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


def load_protein_embedding(embedding_dir: str, protein_id: str) -> torch.Tensor:
    """
    Load a protein embedding from a text file.

    Each line in the file should be in the format: 'index: value1 value2 ... valueN'.

    Args:
        embedding_dir (str): Directory containing the embedding files.
        protein_id (str): ID of the protein to load.

    Returns:
        torch.Tensor: A tensor of shape (sequence_length, embedding_dim) with float32 dtype.
    """
    embeddings: List[List[float]] = []
    with open(f'{embedding_dir}/{protein_id}.embd', 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            _, vec = line.strip().split(':')
            embedding = list(map(float, vec.strip().split()))
            embeddings.append(embedding)
    return torch.tensor(embeddings, dtype=torch.float32)


class PPIDatasetFromFiles(Dataset):
    """
    Dataset for loading protein-protein interaction (PPI) pairs and their labels.

    Each line in the TSV file (no header!) should be in the format: 'protein1\tprotein2\tlabel'.

    Args:
        tsv_file (str): Path to the TSV file containing protein pairs and labels.
        embedding_dir (str): Directory containing protein embedding files.
    """

    def __init__(self, tsv_file: str, embedding_dir: str):
        self.embedding_dir = embedding_dir
        self.data: List[Tuple[str, str, int]] = []
        with open(tsv_file, 'r') as f:
            for line in f:
                p1, p2, label = line.strip().split('\t')
                self.data.append((p1, p2, int(label)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, label = self.data[idx]
        emb1 = load_protein_embedding(self.embedding_dir, p1)
        emb2 = load_protein_embedding(self.embedding_dir, p2)
        return emb1, emb2, torch.tensor(label, dtype=torch.float32)


def padding_puzzler(protein: torch.Tensor, MAX_LENGTH: int = 800, EMBED_DIM: int = 1024) -> torch.Tensor:
    """
    Pads or truncates the input protein tensor to a fixed length and rearranges windows using preloaded permutations.

    Args:
        protein (torch.Tensor): Protein embedding of shape (seq_len, embed_dim).
        MAX_LENGTH (int): Maximum length for padding/truncation. Defaults to 800.
        EMBED_DIM (int): Embedding dimension. Defaults to 1024.

    Returns:
        torch.Tensor: A tensor of shape (8, 53 * EMBED_DIM), representing the "puzzled" version of the protein.
    """
    length = protein.shape[0]
    if length >= MAX_LENGTH:
        padded = protein[:MAX_LENGTH]
    else:
        pad_size = MAX_LENGTH - length
        padding = torch.zeros((pad_size, EMBED_DIM), dtype=protein.dtype)
        padded = torch.cat([protein, padding], dim=0)

    windows = [padded[i:i + 53] for i in range(0, 795, 53)]
    puzzler = np.load('data/permutations.npy')

    puzzeled: List[float] = []
    for i in range(8):
        indices = puzzler[i]
        new_window: List[float] = []
        for index in indices:
            new_window.extend(windows[index])
        puzzeled.append(new_window)

    puzzeled_array = np.array(puzzeled)
    return torch.tensor(puzzeled_array, dtype=torch.float32)


def protein_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    MAX_LENGTH: int = 800,
    EMBED_DIM: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to apply padding and puzzling to each protein in the batch.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): List of (emb1, emb2, label) tuples.
        MAX_LENGTH (int): Maximum length for padding/truncation. Defaults to 800.
        EMBED_DIM (int): Embedding dimension. Defaults to 1024.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batched embeddings and labels.
            - emb1_tensor: (batch_size * 8, 53 * EMBED_DIM)
            - emb2_tensor: (batch_size * 8, 53 * EMBED_DIM)
            - labels_tensor: (batch_size * 8,)
    """
    emb1_list, emb2_list, labels = zip(*batch)
    batch_emb1, batch_emb2 = [], []

    for protein in emb1_list:
        batch_emb1.extend(padding_puzzler(protein, MAX_LENGTH, EMBED_DIM))

    for protein in emb2_list:
        batch_emb2.extend(padding_puzzler(protein, MAX_LENGTH, EMBED_DIM))

    emb1_tensor = torch.stack(batch_emb1)
    emb2_tensor = torch.stack(batch_emb2)

    # Expand labels to match the 8 puzzled versions per sample
    expanded_labels = [item for item in labels for _ in range(8)]
    labels_tensor = torch.tensor(expanded_labels, dtype=torch.float32)

    return emb1_tensor, emb2_tensor, labels_tensor


if __name__ == "__main__":
    # Example usage
    dataset = PPIDatasetFromFiles(
        '/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/pairs/gold_std_tiny_validation.tsv',
        '/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/embd/gold_std'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=protein_collate_fn)

    for emb1, emb2, labels in dataloader:
        print(emb1.shape, emb2.shape, labels.shape)
        break
