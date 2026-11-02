"""
Laptop demo of the AIoT pipeline: image in -> classifier -> virtual sensors -> risk status -> log/chart.

Run:  streamlit run app.py
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from edge_infer import Classifier, assess

st.set_page_config(page_title="Powdery Mildew AIoT", page_icon="🍃", layout="wide")
st.title("🍃 Powdery Mildew Detector – AIoT demo")


@st.cache_resource
def load_model():
    return Classifier("model/model.tflite", "model/labels.json")


if not Path("model/model.tflite").exists():
    st.error("model/model.tflite not found. Run `python train.py` first.")
    st.stop()
clf = load_model()

if "log" not in st.session_state:
    st.session_state.log = []

# ---- sidebar: virtual sensors + settings
with st.sidebar:
    st.header("Virtual sensors")
    temp = st.slider("Temperature (°C)", 5.0, 40.0, 25.0, 0.5)
    hum = st.slider("Humidity (%)", 20, 100, 75)
    st.header("Settings")
    threshold = st.slider("Alert confidence threshold", 0.5, 0.99, 0.75, 0.01)
    use_mqtt = st.checkbox("Publish to MQTT")
    broker = st.text_input("Broker", "localhost", disabled=not use_mqtt)
    topic = st.text_input("Topic", "farm/greenhouse1/leafcam", disabled=not use_mqtt)

# ---- input
src = st.radio("Image source", ["Upload", "Webcam"], horizontal=True)
file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"]) if src == "Upload" else st.camera_input("Take a photo of a leaf")

if file:
    img = Image.open(file).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    t0 = time.time()
    top3 = clf.predict(bgr)
    ms = (time.time() - t0) * 1000
    status, favourable = assess(top3[0], temp, hum, threshold)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(img, use_container_width=True)
    with c2:
        colour = {"OK": "green", "AT_RISK": "orange", "INFECTED": "red", "OTHER_DISEASE": "violet"}[status]
        st.markdown(f"### Status: :{colour}[{status}]")
        st.write(f"**Prediction:** {top3[0][0]}  ({top3[0][1]:.1%})")
        st.write(f"**Inference:** {ms:.0f} ms  |  **Favourable climate:** {'yes' if favourable else 'no'}")
        st.bar_chart(pd.DataFrame({"probability": [p for _, p in top3]}, index=[l for l, _ in top3]))

    record = {"time": time.strftime("%H:%M:%S"), "status": status, "prediction": top3[0][0],
              "confidence": round(top3[0][1], 3), "temp_c": temp, "humidity": hum}
    if not st.session_state.log or st.session_state.log[-1] != record:
        st.session_state.log.append(record)
        if use_mqtt:
            try:
                import paho.mqtt.publish as publish
                publish.single(topic, json.dumps(record), hostname=broker)
                st.toast("Published to MQTT")
            except Exception as e:
                st.warning(f"MQTT failed: {e}")

if st.session_state.log:
    st.subheader("Event log")
    st.dataframe(pd.DataFrame(st.session_state.log).iloc[::-1], use_container_width=True)
