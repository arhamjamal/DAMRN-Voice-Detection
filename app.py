import io
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from damrn import DAMRN


st.set_page_config(
    page_title="DAMRN Audio Deepfake Detection",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ DAMRN Audio Deepfake Detection")
st.caption(
    "Interactive visualization of audio preprocessing "
    "and neural-network feature extraction"
)


@st.cache_resource
def load_model():
    model = DAMRN()
    model.load_state_dict(
        torch.load("damrn.pth", map_location="cpu")
    )
    model.eval()
    return model


def show_heatmap(data, title, xlabel="Time frames",
                 ylabel="Feature / channel"):
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


uploaded = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "ogg"]
)

if uploaded:
    audio_bytes = uploaded.getvalue()

    st.audio(audio_bytes)

    if st.button("🚀 Run DAMRN Detection", type="primary"):

        try:
            progress = st.progress(0)
            status = st.empty()

            # 1. Decode audio
            status.write("**Stage 1/6:** Loading and preprocessing audio...")
            audio, sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=16000
            )

            if len(audio) == 0:
                st.error("The uploaded audio is empty.")
                st.stop()

            duration = len(audio) / sr

            st.write(f"Sample rate: **{sr} Hz**")
            st.write(f"Duration: **{duration:.2f} seconds**")

            fig, ax = plt.subplots(figsize=(10, 2.5))
            times = np.arange(len(audio)) / sr
            ax.plot(times, audio, linewidth=0.6)
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            progress.progress(15)

            # 2. Extract MFCC
            status.write("**Stage 2/6:** Extracting MFCC features...")

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=20
            )

            original_shape = mfcc.shape

            if mfcc.shape[1] >= 50:
                mfcc = mfcc[:, :50]
            else:
                mfcc = np.pad(
                    mfcc,
                    ((0, 0), (0, 50 - mfcc.shape[1])),
                    mode="constant"
                )

            st.write(f"Original MFCC shape: `{original_shape}`")
            st.write(f"Model input features: `{mfcc.shape}`")

            show_heatmap(mfcc, "MFCC Feature Heatmap")
            progress.progress(30)

            # 3. Prepare tensor
            status.write("**Stage 3/6:** Preparing model input...")

            features = torch.tensor(
                mfcc,
                dtype=torch.float32
            ).unsqueeze(0)

            st.write(f"Input tensor shape: `{tuple(features.shape)}`")
            progress.progress(40)

            # 4. Run actual DAMRN
            status.write(
                "**Stage 4/6:** Running CE-Bottleneck, CNN, MSDA and MSFF..."
            )

            model = load_model()

            with torch.no_grad():
                logits, stages = model(
                    features,
                    return_intermediates=True
                )
                probabilities = torch.softmax(logits, dim=1)

            progress.progress(75)

            # 5. Display actual intermediate outputs
            status.write("**Stage 5/6:** Visualizing neural-network features...")

            for stage_name, tensor in stages.items():
                st.subheader(stage_name)

                st.write(
                    f"Tensor shape: `{tuple(tensor.shape)}`"
                )

                # Average across channels for a compact spatial view.
                if tensor.ndim == 4:
                    feature_map = tensor[0].mean(dim=0).cpu().numpy()
                    show_heatmap(
                        feature_map,
                        f"{stage_name} — Mean Feature Map",
                        xlabel="Width",
                        ylabel="Height"
                    )

                    # Show a few real individual channels as well.
                    channel_count = min(4, tensor.shape[1])
                    cols = st.columns(channel_count)

                    for i in range(channel_count):
                        channel = tensor[0, i].cpu().numpy()
                        with cols[i]:
                            st.caption(f"Channel {i + 1}")
                            fig, ax = plt.subplots(figsize=(3, 2))
                            ax.imshow(
                                channel,
                                aspect="auto",
                                origin="lower",
                                cmap="viridis"
                            )
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)

                elif tensor.ndim == 2:
                    st.write(tensor[0].cpu().numpy())

            progress.progress(90)

            # 6. Prediction
            status.write("**Stage 6/6:** Producing final classification...")

            bonafide = probabilities[0, 0].item()
            spoof = probabilities[0, 1].item()

            st.divider()
            st.header("Final Detection Result")

            if spoof >= 0.5:
                st.error("🔴 Prediction: SPOOF")
            else:
                st.success("🟢 Prediction: BONAFIDE")

            col1, col2 = st.columns(2)

            col1.metric(
                "Bonafide probability",
                f"{bonafide * 100:.2f}%"
            )
            col2.metric(
                "Spoof probability",
                f"{spoof * 100:.2f}%"
            )

            st.write("### Model output probabilities")
            st.bar_chart({
                "Probability": {
                    "BONAFIDE": bonafide,
                    "SPOOF": spoof
                }
            })

            progress.progress(100)
            status.success("Analysis complete.")

            st.warning(
                "These are model predictions, not proof of authenticity. "
                "This prototype was trained on only 10 samples, so its "
                "real-world reliability is not established."
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")