import torch
from torch.utils.data import random_split


def get_fixed_split(dataset):

    total = len(dataset)

    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset