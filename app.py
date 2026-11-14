import streamlit as st
from PIL import Image
from src.inference import load_model, predict_image

st.set_page_config(page_title='AURELIS', page_icon='🫁', layout='wide')
st.title('AURELIS')
st.caption('AI-Powered Pneumonia Screening Research Prototype')
st.warning('This project is a prototype and is not medical advice. It is not validated for clinical diagnosis or real-world triage.')
model_path = 'models/best_model.pt'
try:
    model = load_model(model_path)
except Exception as exc:
    st.info('No trained model is installed yet. Train the model using the Colab notebook, then place the exported weights at models/best_model.pt.')
    st.stop()
file = st.file_uploader('Upload a chest X-ray image', type=['png', 'jpg', 'jpeg'])
if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded image', width=420)
    if st.button('Analyze image'):
        with st.spinner('Analyzing image...'):
            result = predict_image(model, image)
        st.metric('Model output', result['label'])
        st.metric('Model confidence', f"{result['confidence']:.1%}")
        st.write(result['message'])
        st.caption('This output is a research-model estimate, not a diagnosis. A qualified healthcare professional must interpret medical images.')
