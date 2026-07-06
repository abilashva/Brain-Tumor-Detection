"""
Brain Tumor Detection - Streamlit App
------------------------------------------------
Upload a brain MRI image and get a real-time tumor / no-tumor prediction
from the CNN trained in train.py.

Run:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = (150, 150)
MODEL_PATH = "brain_tumor_model.h5"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)


def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
    st.title("🧠 Brain Tumor Detection")
    st.write("Upload a brain MRI scan to check for tumor presence.")

    model = get_model()

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"⚠️ Tumor Detected — Confidence: {confidence:.2f}%")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"✅ No Tumor Detected — Confidence: {confidence:.2f}%")

        st.caption(
            "Educational project only — not a substitute for professional medical diagnosis."
        )


if __name__ == "__main__":
    main()
