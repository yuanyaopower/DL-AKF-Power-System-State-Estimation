import torch
import numpy as np
from torch import nn

predicted_indices_large = np.load('index_u_large.npy')


class DualHeadLSTM_large(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_u = nn.Linear(
            hidden_dim,
            len(predicted_indices_large)
        )
        self.head_th = nn.Linear(
            hidden_dim,
            in_dim // 2 - 1
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.drop(h[:, -1])
        return self.head_u(h), self.head_th(h)
