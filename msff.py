import torch
import torch.nn as nn
import torch.nn.functional as F


class MSFF(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, B1, B2, B3):

        P1 = self.conv1(B1)
        P2 = self.conv2(B2)
        P3 = self.conv3(B3)

        P2_up = F.interpolate(
            P2,
            size=P1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        P3_up = F.interpolate(
            P3,
            size=P1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        combined = torch.cat(
            [P1, P2_up, P3_up],
            dim=1
        )

        output = self.fusion(combined)

        return output