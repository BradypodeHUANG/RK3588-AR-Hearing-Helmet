import torch
import torch.nn as nn
from graph import get_adjacency


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution block.

    Graph convolution implemented as Conv2d(1x1) + matmul with adjacency matrix.
    """

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subsets = A.shape[0]

        # One 1x1 conv per adjacency subset
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_subsets)
        ])

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # Register adjacency as buffer (not parameter, but moves with device)
        self.register_buffer('A', A.clone())

    def forward(self, x):
        """x: [N, C, T, V]"""
        res = self.residual(x)

        out = torch.zeros_like(self.convs[0](x))
        for k in range(self.num_subsets):
            x_k = self.convs[k](x)  # [N, C_out, T, V]
            # Graph multiply: [N, C_out, T, V] @ [V, V] -> [N, C_out, T, V]
            out = out + torch.matmul(x_k, self.A[k])

        out = self.bn(out)
        out = self.relu(out + res)
        return out


class MSTCNBlock(nn.Module):
    """Multi-Scale Temporal Convolution block.

    4 parallel branches with different dilation rates.
    """

    def __init__(self, channels, stride=1):
        super().__init__()
        branch_ch = channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=(3, 1),
                      padding=(1, 0), dilation=(1, 1), stride=(stride, 1)),
            nn.BatchNorm2d(branch_ch),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=(3, 1),
                      padding=(2, 0), dilation=(2, 1), stride=(stride, 1)),
            nn.BatchNorm2d(branch_ch),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=(3, 1),
                      padding=(4, 0), dilation=(4, 1), stride=(stride, 1)),
            nn.BatchNorm2d(branch_ch),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(channels, branch_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(branch_ch),
        )

        self.relu = nn.ReLU(inplace=True)

        # Residual
        if stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """x: [N, C, T, V]"""
        res = self.residual(x)
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)
        out = self.relu(out + res)
        return out


class STGCNLayer(nn.Module):
    """One ST-GCN layer = spatial GCN + temporal TCN."""

    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = STGCNBlock(in_channels, out_channels, A)
        self.tcn = MSTCNBlock(out_channels, stride=stride)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return x


class SignLanguageModel(nn.Module):
    """Lightweight ST-GCN + MS-TCN for hand gesture classification.

    Input shape: [N, 6, 60, 26, 2] (C=6, T=60, V=26, M=2 for dual-hand)
    Handles M by folding it into batch, processing each hand through shared GCN,
    then averaging features across hands.
    """

    def __init__(self, num_classes, in_channels=6, num_hands=2):
        super().__init__()
        A = get_adjacency()
        self.num_hands = num_hands

        self.layers = nn.Sequential(
            STGCNLayer(in_channels, 64, A, stride=1),
            STGCNLayer(64, 64, A, stride=2),
            STGCNLayer(64, 128, A, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.shape
        # Fold M into batch: [N*M, C, T, V]
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        x = self.layers(x)  # [N*M, 128, 15, 26]
        x = self.pool(x)    # [N*M, 128, 1, 1]
        x = x.flatten(1)    # [N*M, 128]

        # Unfold M and average: [N, M, 128] -> [N, 128]
        x = x.view(N, M, -1).mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)      # [N, num_classes]
        return x
