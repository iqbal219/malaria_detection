import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Malaria Detection",
    page_icon="🦟",
    layout="centered",
)

MODEL_PATH = Path("outputs/malaria_best_model.keras")
META_PATH = Path("outputs/model_metadata.json")

st.title("🦟 Malaria Detection")
st.caption("Klasifikasi citra sel darah: Uninfected vs Parasitized")

st.warning(
    "Aplikasi ini dibuat untuk pembelajaran/eksperimen machine learning "
    "dan bukan alat diagnosis medis."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. "
            "Jalankan notebook training terlebih dahulu."
        )
    return tf.keras.models.load_model(MODEL_PATH)

def load_metadata():
    default = {
        "class_names": ["Uninfected", "Parasitized"],
        "threshold": 0.5,
        "image_size": [224, 224],
    }
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            return json.load(f)
    return default

metadata = load_metadata()
class_names = metadata["class_names"]
threshold = float(metadata.get("threshold", 0.5))
img_size = tuple(metadata.get("image_size", [224, 224]))

uploaded = st.file_uploader(
    "Upload citra blood smear",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    if st.button("Prediksi", type="primary", use_container_width=True):
        try:
            model = load_model()

            resized = image.resize(img_size)
            x = np.asarray(resized, dtype=np.float32)
            x = np.expand_dims(x, axis=0)

            probability_parasitized = float(
                model.predict(x, verbose=0)[0][0]
            )

            predicted_class = (
                "Parasitized"
                if probability_parasitized >= threshold
                else "Uninfected"
            )

            st.subheader(f"Hasil: {predicted_class}")

            col1, col2 = st.columns(2)
            col1.metric(
                "Parasitized",
                f"{probability_parasitized:.2%}"
            )
            col2.metric(
                "Uninfected",
                f"{1 - probability_parasitized:.2%}"
            )

            st.progress(
                min(max(probability_parasitized, 0.0), 1.0),
                text="Probabilitas Parasitized"
            )

            if predicted_class == "Parasitized":
                st.error(
                    "Model mendeteksi pola yang lebih konsisten dengan kelas Parasitized."
                )
            else:
                st.success(
                    "Model mendeteksi pola yang lebih konsisten dengan kelas Uninfected."
                )

        except Exception as e:
            st.exception(e)
