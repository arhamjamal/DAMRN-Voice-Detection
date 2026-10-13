import torch
import torch.nn as nn


class CE_Bottleneck(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()

        self.spatial_refinement = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            channels,
            channels // reduction
        )

        self.fc2 = nn.Linear(
            channels // reduction,
            channels
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.relu6 = nn.ReLU6()

    def forward(self, x):

        refined = self.spatial_refinement(x)

        pooled = torch.mean(
            refined,
            dim=(2, 3)
        )

        excitation = self.fc1(pooled)
        excitation = self.relu(excitation)

        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        excitation = excitation.unsqueeze(2).unsqueeze(3)

        scaled = refined * excitation

        output = torch.cat(
            [scaled, x],
            dim=1
        )

        output = self.relu6(output)

        return output