import sys
import os
import torch
import librosa

from damrn import DAMRN


# --------------------------------
# Check command-line argument
# --------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("python predict.py <audio_file.wav/mp3/ogg>")
    sys.exit(1)


audio_path = sys.argv[1]


# --------------------------------
# Check file exists
# --------------------------------

if not os.path.isfile(audio_path):
    print("Error: Audio file not found.")
    sys.exit(1)


# --------------------------------
# Check supported format
# --------------------------------

supported_formats = [".wav", ".mp3", ".ogg"]

extension = os.path.splitext(audio_path)[1].lower()

if extension not in supported_formats:
    print("Error: Only WAV, MP3 and OGG files are supported.")
    sys.exit(1)


# --------------------------------
# Load audio
# --------------------------------

audio, sr = librosa.load(
    audio_path,
    sr=16000
)

print()
print("========== AUDIO DEEPFAKE DETECTION ==========")
print()
print("File:", audio_path)
print("Sample Rate:", sr, "Hz")


# --------------------------------
# Extract MFCC
# --------------------------------

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=20
)

print("Original MFCC Shape:", mfcc.shape)


# --------------------------------
# Make exactly 50 frames
# --------------------------------

if mfcc.shape[1] >= 50:

    mfcc = mfcc[:, :50]

else:

    padding = 50 - mfcc.shape[1]

    mfcc = torch.nn.functional.pad(
        torch.tensor(mfcc),
        (0, padding)
    ).numpy()


print("Final MFCC Shape:", mfcc.shape)


# --------------------------------
# Convert to PyTorch tensor
# --------------------------------

features = torch.tensor(
    mfcc,
    dtype=torch.float32
)

features = features.unsqueeze(0)


# --------------------------------
# Load trained DAMRN
# --------------------------------

model = DAMRN()

model.load_state_dict(
    torch.load(
        "damrn.pth",
        map_location="cpu"
    )
)

model.eval()


# --------------------------------
# Prediction
# --------------------------------

with torch.no_grad():

    outputs = model(features)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    bonafide_probability = (
        probabilities[0][0].item()
    )

    spoof_probability = (
        probabilities[0][1].item()
    )


# --------------------------------
# Final prediction
# --------------------------------

if spoof_probability >= 0.5:

    prediction = "SPOOF"

else:

    prediction = "BONAFIDE"


# --------------------------------
# Display result
# --------------------------------

print()
print("Prediction:", prediction)

print(
    f"Bonafide Probability: "
    f"{bonafide_probability * 100:.2f}%"
)

print(
    f"Spoof Probability: "
    f"{spoof_probability * 100:.2f}%"
)

print()
print("==============================================")