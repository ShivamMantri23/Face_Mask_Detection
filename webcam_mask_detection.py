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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        _, cls = model.predict(face_resized, verbose=0)
        class_id = np.argmax(cls)
        label = label_map[class_id]
        confidence = np.max(cls)

        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()