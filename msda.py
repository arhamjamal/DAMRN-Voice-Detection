import torch
import torch.nn as nn
import math


class MSDA(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.key = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.dilation1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1
        )

        self.dilation2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        self.dilation3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3
        )

        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        batch, channels, height, width = Q.shape

        Q = Q.flatten(2).transpose(1, 2)
        K = K.flatten(2).transpose(1, 2)
        V = V.flatten(2).transpose(1, 2)

        attention_scores = torch.matmul(
            Q,
            K.transpose(1, 2)
        )

        attention_scores = (
            attention_scores / math.sqrt(channels)
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attended = torch.matmul(
            attention_weights,
            V
        )

        attended = attended.transpose(1, 2)

        attended = attended.reshape(
            batch,
            channels,
            height,
            width
        )

        F1 = self.dilation1(attended)
        F2 = self.dilation2(attended)
        F3 = self.dilation3(attended)

        combined = torch.cat(
            [F1, F2, F3],
            dim=1
        )

        output = self.fusion(combined)

        return output