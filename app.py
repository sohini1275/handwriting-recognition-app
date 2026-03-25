import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image, ImageOps

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Handwriting Recognition – A–Z",
    layout="centered"
)

st.title("✍️ Handwriting Recognition – A–Z")
st.write("Camera-based handwritten character recognition with AI feedback")
st.write("TensorFlow version:", tf.__version__)

# -------------------------------------------------
# Build model architecture + load weights
# -------------------------------------------------
@st.cache_resource
def build_and_load_model():
    inp = keras.Input(shape=(28, 28, 1), name="input_image")

    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(26, activation="softmax")(x)

    model = keras.Model(inp, out)

    # 🔑 Load weights only (safe across TF versions)
    model.load_weights("model.h5")

    return model

model = build_and_load_model()

st.success("✅ Model loaded successfully (architecture + weights)")
st.write("Input shape:", model.input_shape)
st.write("Output shape:", model.output_shape)

# -------------------------------------------------
# Image preprocessing (MATCH TRAINING)
# -------------------------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("L")          # grayscale
    img = ImageOps.invert(img)      # invert colors
    img = img.resize((28, 28))      # resize
    img_np = np.array(img).astype("float32") / 255.0
    img_np = img_np.reshape(1, 28, 28, 1)
    return img_np

# -------------------------------------------------
# Prediction helper
# -------------------------------------------------
def predict(img_tensor):
    preds = model.predict(img_tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    letter = chr(65 + idx)
    return letter, confidence

# -------------------------------------------------
# Webcam input
# -------------------------------------------------
st.subheader("📷 Webcam Capture")

camera_img = st.camera_input("Write a letter clearly and capture")

if camera_img is not None:
    img = Image.open(camera_img)
    st.image(img, caption="Captured image", width=200)

    if st.button("🔍 Predict from Camera"):
        x = preprocess_image(img)
        letter, conf = predict(x)

        st.markdown("### 🧠 Prediction")
        st.write(f"**Letter:** {letter}")
        st.write(f"**Confidence:** {conf*100:.2f}%")

        if conf < 0.6:
            st.warning("Try writing the letter more clearly or thicker ✍️")
        else:
            st.success("Good handwriting! ✅")

# -------------------------------------------------
# Upload fallback
# -------------------------------------------------
st.divider()
st.subheader("📤 Upload Image (Backup Option)")

uploaded = st.file_uploader(
    "Upload a handwritten letter image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=200)

    if st.button("🔍 Predict from Upload"):
        x = preprocess_image(img)
        letter, conf = predict(x)

        st.markdown("### 🧠 Prediction")
        st.write(f"**Letter:** {letter}")
        st.write(f"**Confidence:** {conf*100:.2f}%")

        if conf < 0.6:
            st.warning("Try writing the letter more clearly or thicker ✍️")
        else:
            st.success("Good handwriting! ✅")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption("Mini Project | Real-time Handwriting Recognition using CNN")
