import os
import librosa
import torch
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):

    def __init__(self, root_dir):

        self.files = []
        self.labels = []

        # Bona-fide = 0
        bonafide_dir = os.path.join(root_dir, "bonafide")

        for file in os.listdir(bonafide_dir):
            if file.endswith(".wav"):
                self.files.append(
                    os.path.join(bonafide_dir, file)
                )
                self.labels.append(0)

        # Spoof = 1
        spoof_dir = os.path.join(root_dir, "spoof")

        for file in os.listdir(spoof_dir):
            if file.endswith(".wav"):
                self.files.append(
                    os.path.join(spoof_dir, file)
                )
                self.labels.append(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        audio_path = self.files[index]
        label = self.labels[index]

        audio, sr = librosa.load(
            audio_path,
            sr=None
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        mfcc = mfcc[:, :50]

        mfcc = torch.tensor(
            mfcc,
            dtype=torch.float32
        )

        return mfcc, label