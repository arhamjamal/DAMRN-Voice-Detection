import librosa

audio, sr = librosa.load(
    "data/small_asvspoof/bonafide/p226_247.wav",
    sr=None
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=20
)

print("Original MFCC shape:", mfcc.shape)

# Take first 50 frames
mfcc_50 = mfcc[:, :50]

print("After selecting 50 frames:", mfcc_50.shape)