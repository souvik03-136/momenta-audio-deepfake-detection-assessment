# model.py

import torch
import torch.nn as nn

class AASIST(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Simple 1D‑conv frontend (RawNet2–style)
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3)
        )
        # Graph‑attention block
        self.graph_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, 1, time)
        """
        x = self.frontend(x)                    # → (batch, 64, T')
        # Prepare for attention: (T', batch, 64)
        x = x.permute(2, 0, 1)
        x, _ = self.graph_attention(x, x, x)
        # Back to (batch, 64, T')
        x = x.permute(1, 2, 0)
        # Global average pool over time
        x = x.mean(dim=2)
        return self.classifier(x)
