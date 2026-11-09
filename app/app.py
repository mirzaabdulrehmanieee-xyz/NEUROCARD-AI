
import sys
from pathlib import Path

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ECGNet

DATA_PATH = PROJECT_ROOT / "data" / "mitbih_test.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "best_model.pth"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ECG Heartbeat Classifier",
    page_icon="❤️",
    layout="wide"
)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = ECGNet()

    state = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


model = load_model()


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_heartbeat(signal):

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(signal_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            logits,
            dim=1
        ).item()

    return predicted_class, probabilities.numpy()[0]


# ==========================================
# 5. MAIN INTERFACE
# ==========================================

st.title("❤️ ECG Heartbeat Classification")

st.write(
    "An AI-based ECG heartbeat classification system "
    "using a trained neural network."
)

st.info(
    "Select a test dataset row or upload a CSV file "
    "containing exactly 187 ECG signal values."
)


# ==========================================
# 6. INPUT SELECTION
# ==========================================

option = st.radio(
    "Choose Input Method",
    [
        "Select Test Dataset Row",
        "Upload CSV"
    ],
    horizontal=True
)

signal = None


# ==========================================
# 7. TEST DATASET ROW SELECTION
# ==========================================

if option == "Select Test Dataset Row":

    if not DATA_PATH.exists():

        st.error("Test dataset file not found!")

    else:

        test_df = pd.read_csv(
            DATA_PATH,
            header=None
        )

        row_number = st.number_input(
            "Select Test Row Number",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1
        )

        signal = test_df.iloc[
            int(row_number), :187
        ].to_numpy(dtype=np.float32)

        st.write(
            f"Selected Test Row: {int(row_number)}"
        )


# ==========================================
# 8. CSV FILE UPLOAD
# ==========================================

else:

    uploaded_file = st.file_uploader(
        "Upload ECG CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file,
                header=None
            )

            if uploaded_df.shape == (1, 187):

                signal = uploaded_df.iloc[0].to_numpy(
                    dtype=np.float32
                )

                st.success(
                    "CSV uploaded successfully!"
                )

            else:

                st.error(
                    "Invalid CSV! File must contain "
                    "exactly one row and 187 numerical values."
                )

        except Exception as e:

            st.error(f"Unable to read CSV: {e}")


# ==========================================
# 9. DISPLAY ECG SIGNAL
# ==========================================

if signal is not None:

    st.divider()

    st.subheader("ECG Heartbeat Signal")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        signal,
        linewidth=1.5
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.set_title("ECG Signal Visualization")

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)


    # ======================================
    # 10. PREDICT HEARTBEAT CLASS
    # ======================================

    predicted_class, probabilities = predict_heartbeat(
        signal
    )

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"Predicted Class: "
            f"{CLASS_NAMES[predicted_class]}"
        )

    with col2:

        st.metric(
            "Prediction Confidence",
            f"{probabilities[predicted_class] * 100:.2f}%"
        )


    # ======================================
    # 11. CLASS PROBABILITY BAR CHART
    # ======================================

    st.subheader("Class Probabilities")

    fig, ax = plt.subplots(figsize=(9, 4))

    bars = ax.bar(
        CLASS_NAMES,
        probabilities * 100
    )

    ax.set_xlabel("Heartbeat Class")
    ax.set_ylabel("Probability (%)")

    ax.set_title(
        "Predicted Probability for Each Class"
    )

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar, probability in zip(
        bars,
        probabilities
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{probability * 100:.2f}%",
            ha="center",
            fontsize=9
        )

    st.pyplot(fig)

    plt.close(fig)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "ECG Heartbeat Classification | "
    "MIT-BIH Dataset | PyTorch Neural Network"
)

st.warning(
    "This is an academic machine learning project "
    "and is not intended for medical diagnosis."
)
