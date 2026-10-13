
import torch
import torch.nn as nn

from ce_bottleneck import CE_Bottleneck
from msda import MSDA
from msff import MSFF


class DAMRN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()

        self.ce = CE_Bottleneck(
            channels=16,
            reduction=4
        )

        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )

        self.conv3 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )

        self.conv4 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )

        self.conv5 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.msda = MSDA(channels=32)
        self.msff = MSFF(channels=32)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 2)
        )

    def forward(self, x, return_intermediates=False):
        stages = {}

        # Input: [batch, 20, 50]
        x = x.unsqueeze(1)

        # Initial convolution
        x = self.conv1(x)
        x = self.relu(x)

        # Channel-Excitation Bottleneck
        x = self.ce(x)
        stages["CE-Bottleneck"] = x

        # CNN feature level B1
        B1 = self.conv2(x)
        B1 = self.relu(B1)
        stages["CNN Feature B1"] = B1

        # CNN feature level B2
        B2 = self.pool(B1)
        B2 = self.conv3(B2)
        B2 = self.relu(B2)
        stages["CNN Feature B2"] = B2

        # CNN feature level B3
        B3 = self.pool(B2)
        B3 = self.conv4(B3)
        B3 = self.relu(B3)
        stages["CNN Feature B3"] = B3

        # Multi-Scale Dilated Attention
        B3_attended = self.msda(B3)
        stages["MSDA Output"] = B3_attended

        # Multi-Scale Feature Fusion
        # The actual MSDA output feeds into MSFF.
        fused = self.msff(B1, B2, B3_attended)
        stages["MSFF Output"] = fused

        # Classification
        output = self.classifier(fused)
        stages["Classifier Logits"] = output

        if return_intermediates:
            return output, stages

        return output
