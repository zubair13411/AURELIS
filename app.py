
from pathlib import Path

import streamlit as st
from PIL import Image

from inference import load_model, predict_image


# AURELIS — Experimental Research Prototype

st.set_page_config(
    page_title="AURELIS",
    page_icon="🫁",
    layout="wide"
)

st.title("AURELIS")
st.subheader("AI-Powered Pneumonia Screening Research Prototype")

st.caption(
    "Experimental chest X-ray image classification "
    "using a ResNet-18 deep learning model."
)

st.warning(
    "IMPORTANT: AURELIS is an experimental research prototype. "
    "It is not clinically validated and must not be used for "
    "medical diagnosis, treatment, or real-world triage."
)

st.divider()


# Model Loading

@st.cache_resource
def get_model():
    return load_model()


model = None

try:
    with st.spinner(
        "Downloading and loading AURELIS model..."
    ):
        model = get_model()

    st.success("Model loaded successfully.")

except Exception as exc:
    st.error("The model could not be loaded.")
    st.caption(f"Technical error: {exc}")


st.divider()


# Image Upload

st.header("Chest X-ray Analysis")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded chest X-ray",
            width=450
        )

        if model is None:
            st.warning(
                "Analysis is unavailable because the "
                "trained model could not be loaded."
            )

        elif st.button("Analyze Image", type="primary"):

            with st.spinner("Processing image..."):
                result = predict_image(model, image)

            st.divider()
            st.header("Research Model Output")

            st.metric(
                "Classification",
                result["label"]
            )

            st.metric(
                "Model Confidence",
                f'{result["confidence"]:.1%}'
            )

            st.info(result["message"])

            st.caption(
                "This is an experimental model output, "
                "not a medical diagnosis. A qualified healthcare "
                "professional must interpret medical images."
            )

    except Exception as exc:
        st.error("Unable to process the uploaded image.")
        st.caption(f"Technical error: {exc}")


# Footer

st.divider()

st.caption(
    "AURELIS | AI Research Project | ResNet-18"
)

st.caption(
    "For educational and research purposes only. "
    "Not intended for clinical use."
)
