import torch
import numpy as np

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from dataset import AudioDeepfakeDataset
from damrn import DAMRN


# -------------------------
# Dataset
# -------------------------

dataset = AudioDeepfakeDataset(
    "data/small_asvspoof"
)

total = len(dataset)

train_size = int(0.6 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)


# -------------------------
# Load trained model
# -------------------------

model = DAMRN()

model.load_state_dict(
    torch.load(
        "damrn.pth",
        map_location="cpu"
    )
)

model.eval()


# -------------------------
# Collect predictions
# -------------------------

true_labels = []
predicted_labels = []
spoof_scores = []


with torch.no_grad():

    for features, labels in test_loader:

        outputs = model(features)

        # Probability of spoof class
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        spoof_probability = probabilities[:, 1]

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions.numpy()
        )

        spoof_scores.extend(
            spoof_probability.numpy()
        )


# Convert to NumPy
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
spoof_scores = np.array(spoof_scores)


# -------------------------
# Accuracy
# -------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# -------------------------
# F1 Score
# -------------------------

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# -------------------------
# AUC
# -------------------------

try:

    auc = roc_auc_score(
        true_labels,
        spoof_scores
    )

except ValueError:

    auc = float("nan")


# -------------------------
# Print results
# -------------------------

print()
print("========== DAMRN EVALUATION ==========")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)

print()
print("True Labels     :", true_labels)
print("Predicted Labels:", predicted_labels)
print("Spoof Scores    :", spoof_scores)