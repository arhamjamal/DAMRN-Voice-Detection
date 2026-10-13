import librosa
import librosa.display
import matplotlib.pyplot as plt

audio, sr = librosa.load(
    "data/small_asvspoof/bonafide/p226_247.wav",
    sr=None
)

plt.figure(figsize=(12, 4))

librosa.display.waveshow(audio, sr=sr)

plt.title("Waveform - Bona-fide Audio")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.show()