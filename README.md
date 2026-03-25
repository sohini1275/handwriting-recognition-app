# ✍️ Real-Time Handwriting Recognition (A–Z)

A deep learning-based web application that recognizes handwritten English alphabets (A–Z) using a Convolutional Neural Network (CNN) with real-time camera input.

---

## 🚀 Features
- 📷 Webcam-based handwritten input
- 🔍 Real-time character prediction
- 📊 Confidence score display
- 💡 Feedback for handwriting improvement
- 📤 Image upload support (backup)

---

## 🧠 Model Details
- CNN trained on A–Z handwritten dataset (~370k samples)
- Input: 28×28 grayscale images
- Accuracy: ~99%

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
