import streamlit as st
from PIL import Image
from inference import load_session, predict_image

st.set_page_config(
    page_title="AURELIS | Research Prototype",
    page_icon="🩺",
    layout="centered"
)

st.title("AURELIS")
st.subheader("Experimental Chest X-ray Classification")

st.warning(
    "RESEARCH PROTOTYPE ONLY: This tool is not clinically validated. "
    "It must not be used for diagnosis, treatment, or medical triage. "
    "Always consult qualified healthcare professionals."
)

st.markdown(
    "Upload a chest X-ray image to explore the experimental "
    "machine-learning model output."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-ray image",
        use_container_width=True
    )

    if st.button("Run Experimental Analysis", type="primary"):
        try:
            with st.spinner("Loading model and processing image..."):
                session = load_session()
                result = predict_image(session, image)

            st.subheader("Experimental Model Output")
            st.write("**Result:**", result["label"])
            st.write(
                "**Model confidence:**",
                f'{result["confidence"] * 100:.2f}%'
            )

            st.info(result["message"])

        except Exception as error:
            st.error("Unable to process the image.")
            st.exception(error)

st.divider()

st.caption(
    "AURELIS | Educational and research purposes only. "
    "Not a medical device or diagnostic system."
)
