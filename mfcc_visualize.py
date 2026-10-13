import librosa
import librosa.display
import matplotlib.pyplot as plt

audio, sr = librosa.load(
    "data/small_asvspoof/bonafide/p226_247.wav",
    sr=None
)

# Extract 20 MFCC coefficients
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=20
)

# Keep 50 frames
mfcc_50 = mfcc[:, :50]

print("MFCC shape:", mfcc_50.shape)

# Plot MFCC
plt.figure(figsize=(10, 5))

librosa.display.specshow(
    mfcc_50,
    x_axis="time"
)

plt.colorbar()

plt.title("20 × 50 MFCC - Bona-fide Audio")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")

plt.tight_layout()
plt.show()