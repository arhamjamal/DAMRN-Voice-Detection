from torch.utils.data import random_split
from dataset import AudioDeepfakeDataset


# Load complete dataset
dataset = AudioDeepfakeDataset(
    "data/small_asvspoof"
)

# Total samples
total = len(dataset)

# 60% train
train_size = int(0.6 * total)

# 20% validation
val_size = int(0.2 * total)

# Remaining samples for test
test_size = total - train_size - val_size


# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)


print("Total samples:", total)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))