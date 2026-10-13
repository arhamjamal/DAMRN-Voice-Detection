import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AudioDeepfakeDataset
from ce_cnn import CECNN
from split_utils import get_fixed_split


# -------------------------
# Dataset
# -------------------------

dataset = AudioDeepfakeDataset(
    "data/small_asvspoof"
)

train_dataset, val_dataset, test_dataset = get_fixed_split(
    dataset
)


# -------------------------
# DataLoaders
# -------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = CECNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Training
# -------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total_samples = 0

    for features, labels in train_loader:

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    train_accuracy = (
        correct / total_samples
    ) * 100


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in val_loader:

            outputs = model(features)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = (
        val_correct / val_total
    ) * 100


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# -------------------------
# Test
# -------------------------

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for features, labels in test_loader:

        outputs = model(features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)


test_accuracy = (
    test_correct / test_total
) * 100


print()
print("Test Accuracy:", test_accuracy, "%")


# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "ce_cnn.pth"
)

print("Model saved as ce_cnn.pth")