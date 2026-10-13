import numpy as np
from sklearn.metrics import roc_curve


def calculate_eer(labels, scores):

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    fnr = 1 - tpr

    # Find the point where FAR and FRR are closest
    difference = np.abs(fpr - fnr)

    index = np.argmin(difference)

    eer = (fpr[index] + fnr[index]) / 2

    return eer


# Example test
labels = np.array([0, 1])
scores = np.array([0.0688, 0.9999])

eer = calculate_eer(
    labels,
    scores
)

print("EER:", eer)
print("EER (%):", eer * 100)