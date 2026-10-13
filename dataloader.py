from torch.utils.data import DataLoader
from dataset import AudioDeepfakeDataset


# Create dataset
dataset = AudioDeepfakeDataset(
    "data/small_asvspoof"
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Get one batch
features, labels = next(iter(dataloader))

print("Batch feature shape:", features.shape)
print("Batch labels shape:", labels.shape)
print("Labels:", labels)