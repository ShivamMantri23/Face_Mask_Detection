import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mask_detector.keras")

label_map = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect"
}

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Face Mask Detection", layout="centered")
st.title("😷 Face Mask Detection System")

st.write("Choose an input method:")

option = st.radio(
    "Select Input Type",
    ("Upload Image", "Use Webcam")
)

def detect_and_draw(img_rgb):
    """Detect face using Haar and classify mask"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img_rgb[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        _, cls = model.predict(face_resized, verbose=0)
        class_id = np.argmax(cls)
        confidence = np.max(cls)

        label = label_map[class_id]

        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)

        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            img_rgb,
            f"{label} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return img_rgb

if option == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        img_bytes = uploaded.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = detect_and_draw(img_rgb)

        st.image(result, caption="Prediction Result", use_column_width=True)

if option == "Use Webcam":
    img_file = st.camera_input("Take a photo using webcam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = detect_and_draw(img_rgb)

        st.image(result, caption="Webcam Prediction", use_column_width=True)