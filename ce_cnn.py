import torch
import torch.nn as nn

from ce_bottleneck import CE_Bottleneck


class CECNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # CE-Bottleneck
        self.ce = CE_Bottleneck(
            channels=16,
            reduction=4
        )

        # After CE-Bottleneck:
        # 16 + 16 = 32 channels
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        # [batch, 20, 50]
        x = x.unsqueeze(1)

        # [batch, 1, 20, 50]
        x = self.conv1(x)
        x = self.relu(x)

        # [batch, 16, 20, 50]
        x = self.ce(x)

        # [batch, 32, 20, 50]
        x = self.conv2(x)
        x = self.relu(x)

        # [batch, 32, 10, 25]
        x = self.pool(x)

        # [batch, 32, 5, 12]
        x = self.pool(x)

        # Classification
        x = self.classifier(x)

        return x


# Test the model
model = CECNN()

x = torch.randn(4, 20, 50)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)