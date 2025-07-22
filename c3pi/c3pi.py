import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvDenseBlock(nn.Module):
    """
    A convolutional + dense block consisting of:
    - Conv1D
    - BatchNorm
    - ReLU
    - Flatten
    - Dense (Linear)
    - Dropout

    The dense layer is initialized dynamically based on input shape.
    """

    def __init__(
        self, 
        in_channels: int, 
        conv_kernel_size: int, 
        conv_stride: int, 
        dense_units: int, 
        dropout_rate: float = 0.5
    ):
        super(ConvDenseBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=64,
                              kernel_size=conv_kernel_size, stride=conv_stride)
        self.bn = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self._dense_units = dense_units
        self.fc = None  # Linear layer defined dynamically

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # From (B, L, C) to (B, C, L)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.flatten(x)

        # Lazy initialization of linear layer
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self._dense_units).to(x.device)

        x = self.fc(x)
        x = self.dropout(x)
        return x


class PPIModelBranch(nn.Module):
    """
    One branch of the full PPI model that contains several ConvDenseBlocks
    with different kernel sizes and strides to capture features at multiple scales.
    """

    def __init__(
        self, 
        in_channels: int, 
    ):
        super(PPIModelBranch, self).__init__()
        kernel_sizes = [795, 400, 200, 100, 50, 20]
        strides = [1, 200, 100, 50, 25, 10]
        dense_units = [16, 32, 64, 128, 256, 512]

        self.blocks = nn.ModuleList([
            ConvDenseBlock(in_channels, k, s, d)
            for k, s, d in zip(kernel_sizes, strides, dense_units)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [block(x) for block in self.blocks]


class FullPPIModel(nn.Module):
    """
    Full PPI prediction model with two parallel branches processing protein inputs.
    Each layer’s output from both branches is fused and aggregated before classification.
    """

    def __init__(
        self, 
        input_features_dim: int
    ):
        super(FullPPIModel, self).__init__()
        self.branch1 = PPIModelBranch(input_features_dim)
        self.branch2 = PPIModelBranch(input_features_dim)

        dense_units = [16, 32, 64, 128, 256, 512]
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Dropout(0.5))
            for d in dense_units
        ])

        self.final = nn.Sequential(
            nn.Linear(sum(dense_units), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        features1 = self.branch1(input1)
        features2 = self.branch2(input2)

        concatenated_features = []
        for idx, (f1, f2) in enumerate(zip(features1, features2)):
            concat = torch.cat((f1, f2), dim=1)
            fused = self.fusion_layers[idx](concat)
            concatenated_features.append(fused)

        merged = torch.cat(concatenated_features, dim=1)
        out = self.final(merged)
        return out


if __name__ == "__main__":
    from torchviz import make_dot
    model = FullPPIModel(input_features_dim=1024)
    input1 = torch.randn(2, 795, 1024)  # (batch_size, sequence_length, embedding_dim)
    input2 = torch.randn(2, 795, 1024)

    output = model(input1, input2)
    make_dot(output, params=dict(model.named_parameters())).render("ppi_model", format="png")
