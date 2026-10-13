import torch
from torch.utils.data import random_split

from dataset import AudioDeepfakeDataset


# Load dataset
dataset = AudioDeepfakeDataset(
    "data/small_asvspoof"
)

# Dataset sizes
total = len(dataset)

train_size = int(0.6 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size


# Fixed random generator
generator = torch.Generator().manual_seed(42)


# Create reproducible split
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)


print("========== FIXED DATASET SPLIT ==========")

print("Total samples      :", len(dataset))
print("Training samples   :", len(train_dataset))
print("Validation samples :", len(val_dataset))
print("Testing samples    :", len(test_dataset))

print()
print("Train indices:", train_dataset.indices)
print("Validation indices:", val_dataset.indices)
print("Test indices:", test_dataset.indices)