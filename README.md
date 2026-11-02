# Powdery Mildew Detection – AIoT Project
Cherry · Squash · Cucumber · Apple

```
 Leaf camera ─┐
              ├─► Edge node (Raspberry Pi, TFLite) ─► MQTT ─► Node-RED / Grafana / phone alert
 DHT22 temp/RH┘        classify + risk logic
```

## 1. Get the data
```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json
mkdir raw && cd raw
kaggle datasets download -d codeinstitute/cherry-leaves --unzip -p cherry-leaves
kaggle datasets download -d abdallahalidev/plantvillage-dataset --unzip -p plantvillage-dataset
kaggle datasets download -d kaushigihanml/cucumber-leaf-disease-dataset --unzip -p cucumber-leaf-disease-dataset
kaggle datasets download -d aibuzz/apple-leaf-disease-powdery-mildew --unzip -p apple-leaf-disease-powdery-mildew
cd ..
```

## 2. Merge + split
```bash
pip install -r requirements.txt
python prepare_dataset.py --cherry raw/cherry-leaves --plantvillage raw/plantvillage-dataset \
    --cucumber raw/cucumber-leaf-disease-dataset --apple raw/apple-leaf-disease-powdery-mildew \
    --out data --verbose
```
Check the printed class table. Expected classes (some depend on folder names in the cucumber/apple sets):
`apple_healthy, apple_powdery_mildew, cherry_healthy, cherry_powdery_mildew, squash_powdery_mildew, cucumber_healthy, cucumber_powdery_mildew, cucumber_downy_mildew`.
If a class is missing, run with `--verbose`, see which folders were skipped, and adjust keywords in `prepare_dataset.py`.

## 3. Train (GPU recommended, Colab works)
```bash
python train.py --data data --out model
```
Outputs `model/model.tflite` (~3 MB) and `model/labels.json`, plus a per-class report and a
"powdery mildew alert" precision/recall (the metric that matters for the IoT alert).

## 4a. Run on your laptop (no hardware needed)
```bash
pip install -r requirements.txt
streamlit run app.py          # upload or webcam image + virtual temp/humidity sliders
```
or the command line version with a live webcam loop and simulated sensors:
```bash
python edge_infer.py --image leaf.jpg --no-mqtt --simulate
python edge_infer.py --camera 0 --no-mqtt --simulate --interval 5
```
Optional MQTT on the laptop: install Mosquitto, run `mosquitto -v`, tick "Publish to MQTT" in the app
(or drop `--no-mqtt`), and watch with `mosquitto_sub -t "farm/#" -v`.
Later, the same `model.tflite` and `edge_infer.py` move to a Raspberry Pi unchanged.

## 4b. Run on the edge (Raspberry Pi, later)
```bash
# on Raspberry Pi
pip install tflite-runtime opencv-python-headless paho-mqtt adafruit-circuitpython-dht
python edge_infer.py --image test_leaf.jpg --no-mqtt
python edge_infer.py --camera 0 --broker <broker-ip> --dht-pin D4
```
MQTT topic `farm/greenhouse1/leafcam` receives JSON:
```json
{"node":"node-01","status":"INFECTED","prediction":"cherry_powdery_mildew","confidence":0.97,
 "temp_c":24.5,"humidity":82.0,"favourable_conditions":true,"latency_ms":41.3}
```
Status values: `OK`, `AT_RISK` (humid 20–30 °C, no visual signs yet), `INFECTED`, `OTHER_DISEASE` (downy mildew).
Set up Mosquitto as the broker and Node-RED or Grafana to display it.

## Known limitations (be upfront about these in your report)
- **Squash has no healthy class** in PlantVillage, so the model can learn "squash leaf = mildew". Add healthy squash images (your own or another dataset) before trusting squash results.
- **Lab vs field gap.** PlantVillage and Cherry Leaves are mostly clean, single-leaf photos on plain backgrounds. Expect accuracy to drop on real greenhouse frames. Collect 100–200 of your own camera images per class and fine-tune on them.
- **Near-duplicate leakage.** PlantVillage has several photos of the same leaf; a random split inflates test accuracy. Exact duplicates are removed, but near-duplicates are not. Judge the model on your own field images.
- Mount the camera at a fixed distance with consistent lighting, and frame one leaf.
- The risk logic thresholds (20–30 °C, ≥70 % RH) are a simple heuristic, so tune them per crop.

## Extensions
ESP32-CAM streaming to a Pi/server instead of on-device inference · Grad-CAM overlays · int8 full quantisation for Coral/ESP32-S3 · relay output to trigger ventilation or a sprayer.
