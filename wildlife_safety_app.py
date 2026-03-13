import streamlit as st

st.set_page_config(page_title="Wildlife Threat Detection", layout="centered")

import numpy as np
import librosa
import serial
import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import tempfile
import time
import os

# ------------------------ SERIAL CONNECTION ------------------------
@st.cache_resource
def connect_esp32(port='COM8', baudrate=115200):
    try:
        esp = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)
        return esp
    except serial.SerialException:
        return None

esp32 = connect_esp32()


yolo_model = YOLO("D:/Yolov5/runs/detect/train28/weights/best.pt")
cnn_model = load_model("audio_classification_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler2.pkl")


threat_classes = {"cheetah", "bear", "elephant", "lion"}
non_threat_classes = {"human"}


st.title("🛡 Wildlife Threat Detection System")

tab1, tab2, tab3 = st.tabs([
    "📷 Image Detection", "🎧 Audio Detection", "📹 Video Detection"
])

# ------------------------ IMAGE DETECTION ------------------------
with tab1:
    st.subheader("Upload Image")
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img")

    if image_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_path, 'wb') as f:
            f.write(image_file.read())

        st.image(temp_path, caption="Uploaded Image", use_container_width=True)
        st.write("🔍 Detecting...")

        results = yolo_model(temp_path)
        boxes = results[0].boxes
        class_names = results[0].names
        detected_classes = set()
        threshold = 0.6

        for box in boxes:
            conf = float(box.conf)
            cls_index = int(box.cls)
            class_name = class_names[cls_index].lower()

            if conf < threshold:
                continue

            detected_classes.add(class_name)

            if class_name in threat_classes:
                st.warning(f"⚠ Danger Detected: {class_name.capitalize()}! Confidence: {conf:.2f}")
                if esp32:
                    esp32.write(b'1\n')
            elif class_name in non_threat_classes:
                st.info(f"Detected: {class_name.capitalize()} – No threat. Confidence: {conf:.2f}")
            else:
                st.info(f"Detected: {class_name.capitalize()} – Unknown class. Confidence: {conf:.2f}")

        if not detected_classes:
            st.success("✅ No valid threats or known classes detected.")

# ------------------------ AUDIO DETECTION ------------------------
with tab2:
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose a WAV file", type=["wav"], key="audio")

    if audio_file:
        st.audio(audio_file)

        y, sr = librosa.load(audio_file, sr=22050, duration=5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)
        mfcc_scaled = scaler.transform(mfcc)
        mfcc_scaled = mfcc_scaled[..., np.newaxis]

        pred = cnn_model.predict(mfcc_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(pred)])[0]

        st.success(f"Predicted: {predicted_label}")

        if predicted_label.lower() in threat_classes:
            st.warning(f"⚠ Danger Detected: {predicted_label.capitalize()} in Audio!")
            if esp32:
                esp32.write(b'1\n')
        elif predicted_label.lower() == "human":
            st.info("✅ No threat detected.")

# ------------------------ VIDEO DETECTION ------------------------
with tab3:
    st.subheader("Upload Video for Detection")
    video_file = st.file_uploader("Choose an MP4 file", type=["mp4"], key="video")

    if video_file:
        st.video(video_file)


        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())

        st.write("🔍 Processing Video... Please wait.")

        cap = cv2.VideoCapture(temp_video_path)

        out_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        detected_classes = set()
        alert_triggered = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            boxes = results[0].boxes
            class_names = yolo_model.names

            for box in boxes:
                conf = float(box.conf)
                cls_index = int(box.cls)
                class_name = class_names[cls_index].lower()

                if conf < 0.3:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if class_name not in detected_classes:
                    detected_classes.add(class_name)
                    st.write(f"Detected: {class_name.capitalize()}")


                if class_name in threat_classes and not alert_triggered:
                    st.warning(f"⚠ Danger Detected: {class_name.capitalize()}!")
                    if esp32:
                        esp32.write(b'1\n')
                    alert_triggered = True

         
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))

            out.write(frame)

        cap.release()
        if out:
            out.release()

        st.success("✅ Video processing completed.")


        with open(out_path, 'rb') as f:
            video_bytes = f.read()
            st.video(video_bytes)
