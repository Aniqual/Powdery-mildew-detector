"""
Edge node (runs on a laptop too: use --simulate and --no-mqtt).

Camera -> TFLite classifier -> (optional) DHT22 temp/humidity -> risk logic -> MQTT JSON.

Examples:
  python edge_infer.py --image leaf.jpg --no-mqtt             # quick test, no hardware
  python edge_infer.py --camera 0 --broker 192.168.1.10       # live loop
  python edge_infer.py --camera 0 --broker 192.168.1.10 --dht-pin D4
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # laptop / desktop: use full TensorFlow
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


class Classifier:
    def __init__(self, model_path, labels_path):
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.labels = json.loads(Path(labels_path).read_text())
        self.size = self.inp["shape"][1]

    def predict(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb.astype(np.float32), 0)  # raw 0..255, scaling is inside the model
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out["index"])[0]
        order = probs.argsort()[::-1][:3]
        return [(self.labels[i], float(probs[i])) for i in order]


def make_dht(pin_name):
    if not pin_name:
        return None
    try:
        import adafruit_dht
        import board
        return adafruit_dht.DHT22(getattr(board, pin_name))
    except Exception as e:
        print(f"[warn] DHT22 unavailable: {e}")
        return None


_sim = {"t": 25.0, "h": 65.0}


def read_env(dht, simulate=False):
    if simulate:  # virtual sensor for laptop testing: slow random walk
        import random
        _sim["t"] = min(35, max(10, _sim["t"] + random.uniform(-0.5, 0.5)))
        _sim["h"] = min(95, max(30, _sim["h"] + random.uniform(-2, 2)))
        return round(_sim["t"], 1), round(_sim["h"], 1)
    if dht is None:
        return None, None
    try:
        return dht.temperature, dht.humidity
    except Exception:  # DHT sensors throw transient read errors
        return None, None


def assess(top, temp, hum, threshold):
    """Combine vision result with microclimate. Powdery mildew favours ~20-30C and high humidity."""
    label, conf = top
    favourable = temp is not None and hum is not None and 20 <= temp <= 30 and hum >= 70
    if label.endswith("powdery_mildew") and conf >= threshold:
        return "INFECTED", favourable
    if label.endswith("downy_mildew") and conf >= threshold:
        return "OTHER_DISEASE", favourable
    if favourable:
        return "AT_RISK", True
    return "OK", False


def connect_mqtt(broker, port):
    import paho.mqtt.client as mqtt
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt >= 2.0
    except AttributeError:
        c = mqtt.Client()
    c.connect(broker, port, 60)
    c.loop_start()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/model.tflite")
    ap.add_argument("--labels", default="model/labels.json")
    ap.add_argument("--image", help="classify one image and exit")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=30, help="seconds between captures")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="farm/greenhouse1/leafcam")
    ap.add_argument("--node-id", default="node-01")
    ap.add_argument("--dht-pin", help="board pin name for DHT22, e.g. D4")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="fake temp/humidity (no DHT22 needed)")
    a = ap.parse_args()

    clf = Classifier(a.model, a.labels)
    dht = make_dht(a.dht_pin)
    mq = None if a.no_mqtt else connect_mqtt(a.broker, a.port)
    cap = None if a.image else cv2.VideoCapture(a.camera)

    while True:
        if a.image:
            frame = cv2.imread(a.image)
            if frame is None:
                raise SystemExit(f"cannot read {a.image}")
        else:
            ok, frame = cap.read()
            if not ok:
                print("[warn] camera read failed")
                time.sleep(5)
                continue

        t0 = time.time()
        top3 = clf.predict(frame)
        latency = (time.time() - t0) * 1000
        temp, hum = read_env(dht, a.simulate)
        status, favourable = assess(top3[0], temp, hum, a.threshold)

        msg = {
            "node": a.node_id, "ts": int(time.time()),
            "status": status, "prediction": top3[0][0], "confidence": round(top3[0][1], 3),
            "top3": [{"label": l, "p": round(p, 3)} for l, p in top3],
            "temp_c": temp, "humidity": hum, "favourable_conditions": favourable,
            "latency_ms": round(latency, 1),
        }
        print(json.dumps(msg))
        if mq:
            mq.publish(a.topic, json.dumps(msg), qos=1)
            if status == "INFECTED":  # also send a small alert topic for dashboards/notifications
                mq.publish(a.topic + "/alert", json.dumps(msg), qos=1, retain=True)

        if a.image:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
