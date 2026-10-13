import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # Reduce size
            nn.MaxPool2d(kernel_size=2),

            # Second convolution
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # Reduce size again
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 5 * 12,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        # Add channel dimension
        x = x.unsqueeze(1)

        x = self.features(x)

        x = self.classifier(x)

        return x


# Create model
model = BaselineCNN()

print(model)