# Driver-Drowsiness-Detection-System_TEAM04

## Edge AI using Sony IMX500 and Raspberry Pi

### 1. Introduction
This project implements a **Driver Drowsiness Detection System** using a **Sony IMX500 AI camera** and **Raspberry Pi**. The system performs real-time driver state detection and provides visual warnings and audio alerts to enhance road safety.

The solution demonstrates a complete embedded AI pipeline, including:
- Dataset preparation
- YOLO model training
- Model export to IMX format
- Conversion to RPK
- Real-time deployment on Raspberry Pi

---

### 2. Motivation and Concept
Driver drowsiness is a significant contributor to traffic accidents. Modern vehicles increasingly rely on Driver Monitoring Systems (DMS) to detect unsafe driver behavior.

This project aims to:
- Detect driver alertness states in real time
- Run inference fully on-device (edge AI)
- Use a lightweight model suitable for embedded hardware
- Provide immediate driver feedback through visual and audio alerts

The project aligns with automotive AI applications and satisfies all academic project requirements.

---

### 3. Detected Driver States
The trained object detection model recognizes the following classes:
- **awake** – Driver is alert
- **sleeping** – Driver is drowsy / eyes closed
- **sunglass_detected** – Sunglasses detected (reduced eye visibility)

---

### 4. Hardware and Software Setup

#### Hardware:
- Raspberry Pi (64-bit OS)
- Sony IMX500 AI Camera
- Speaker / Headphones (audio alerts)

#### Software:
- Python 3.11
- OpenCV
- Picamera2
- Ultralytics YOLO
- Sony IMX500 tools
- Pygame (audio output)

---

### 5. Project Folder Structure (Actual ZIP Layout)

```plaintext
DROWSINESS_PROJECT_SUBMISSION/
├── data/
│   ├── train/
│   └── validation/
├── imx500/
│   ├── audio/
│   ├── best_imx_model/
│   ├── best_imx.onnx
│   ├── best_imx.pbtxt
│   ├── best_imx_MemoryReport.json
│   ├── dnnParams.html
│   ├── labels.txt
│   ├── packerOut.zip
│   ├── imx500_demo_v1.py
│   └── imx500_demo_v2.py
├── train_model/
│   ├── train.py
│   ├── yolo_export.py
│   ├── train/
│   ├── weights/
│   ├── best.pt
│   ├── last.pt
│   ├── best_imx_model/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── BoxPR_curve.png
│   ├── results.csv
│   ├── train_batch*.jpg
│   ├── val_batch*_pred.jpg
│   ├── classes.txt
│   ├── notes.json
│   ├── prepare_training_data.py
│   ├── yolo_config.yaml
└── README.md
```

---

### 6. End-to-End Pipeline Overview
The system follows the official **IMX500 edge-AI workflow**:
- Dataset preparation
- Data annotation
- YOLO model training
- Model evaluation
- Export to IMX format
- Conversion to RPK
- Real-time deployment

---

### 7. Dataset Preparation

#### 7.1 Data Collection
- Images captured manually
- Fixed camera position
- Multiple lighting conditions and head poses
- Real human facial data

#### 7.2 Dataset Structure
```plaintext
data/
├── train/
└── validation/
```
Each image has a corresponding YOLO-format label file.

---

### 8. Data Annotation

#### Annotation Tool:
- **Label Studio**

#### Annotation Format:
- YOLO bounding boxes
- One face per image

#### Class Mapping:
- **0** awake
- **1** sleeping
- **2** sunglass_detected

---

### 9. Model Training (YOLO)

#### 9.1 Model Selection
**YOLOv11 Nano** - Lightweight and suitable for embedded deployment.

#### 9.2 Training Configuration:
- Image size: 640 × 640
- Epochs: 100
- Batch size: 16
- Optimizer: YOLO default

#### Training Script:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="yolo_config.yaml", epochs=100, imgsz=640, batch=16)
```

---

### 10. Evaluation and Results

#### 10.1 Quantitative Metrics:
- Precision
- Recall
- mAP@0.5
- Loss curves

#### 10.2 Training Behaviour:
- Training and validation losses decrease steadily.
- No major overfitting observed.
- Stable convergence across epochs.

#### 10.3 Confusion Matrix Analysis:
- Strong diagonal dominance for awake and sleeping.
- Minor confusion between sleeping and sunglass_detected.
- Low background false positives.

#### 10.4 Qualitative Evaluation:
- Accurate bounding box localization.
- Stable detection across head poses.
- Good generalization on validation data.

---

### 11. Export to IMX500 Format
The trained YOLO model is exported to IMX format.

#### Export Command:
```bash
python yolo_export.py --init_model train/weights/best.pt --export_format imx --export_only --int8_weights
```

#### Generated Files:
```plaintext
imx500/best_imx_model/
├── packerOut.zip
├── labels.txt
├── best_imx.onnx
├── best_imx.pbtxt
└── best_imx_MemoryReport.json
```

---

### 12. Conversion to RPK
On Raspberry Pi:
```bash
sudo apt install imx500-all imx500-package -i packerOut.zip -o network.rpk
```

---

### 13. Real-Time Deployment
Run Command:
```bash
python imx500_demo_v2.py --model network.rpk --labels labels.txt --threshold 0.6
```

#### Runtime Features:
- Bounding box visualization
- Status banner overlay
- Sleeping timer display
- Audio alerts
- Danger escalation after 5 seconds of continuous sleeping

---

### 14. Limitations
- Limited dataset size
- No infrared camera (night conditions)
- Sunglasses reduce eye visibility
- Single-camera setup

---

### 15. Possible Improvements
- Eye Aspect Ratio (EAR) based detection
- Infrared camera integration
- Temporal models (LSTM / GRU)
- CAN / Car2X vehicle integration
- Multi-camera fusion

---

### 16. Conclusion
This project demonstrates a complete embedded AI pipeline for driver monitoring, from dataset creation to real-time on-device inference using Sony IMX500. The system meets all academic and technical requirements and demonstrates the feasibility of edge-AI-based driver drowsiness detection for automotive applications.

---

### Author
- Yutsav Hari Bhagat – 12500192
- Deep Bharatbhai Savaliya - 12501180  
Master’s Student – Automotive / AI Systems
