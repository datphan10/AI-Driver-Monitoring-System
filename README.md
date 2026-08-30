# AI-Based Driver Monitoring System

A real-time **Driver Monitoring System (DMS)** based on computer vision and deep learning.

The project uses facial detection and a custom 98-point facial landmark model to analyze important facial features of the driver. Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose estimation are used to support driver-state monitoring and real-time warning.

---

## Demo

### 98-Point Facial Landmark Detection

| Input Image | Landmark Detection Result |
|---|---|
| ![](assets/test.jpg) | ![](assets/output.jpg) |

The system first detects the face and then predicts **98 facial landmark points** for further driver-state analysis.

---

## Overview

Driver behavior is an important factor in road safety. A Driver Monitoring System can use a camera to continuously analyze facial information and identify visual cues related to the driver's state.

This project implements a vision-based processing pipeline consisting of:

1. Face detection
2. Face preprocessing
3. 98-point facial landmark prediction
4. Eye Aspect Ratio (EAR) calculation
5. Mouth Aspect Ratio (MAR) calculation
6. Head pose estimation
7. Real-time driver-state analysis
8. Visual and audio warning

The facial landmark model is based on a modified **MobileNetV2** architecture and is trained using the **WFLW facial landmark dataset**.

---

## System Pipeline

```text
Camera / Input Image
        |
        v
+----------------------+
|    Face Detection    |
|      MediaPipe       |
+----------------------+
        |
        v
+----------------------+
|   Face Preprocessing |
| Grayscale + 112x112  |
+----------------------+
        |
        v
+----------------------+
|  Modified MobileNetV2|
+----------------------+
        |
        v
+----------------------+
| 98 Facial Landmarks  |
+----------------------+
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+---------------+   +---------------+   +---------------+
|      EAR      |   |      MAR      |   |   SolvePnP    |
| Eye Analysis  |   | Mouth Analysis|   |   Head Pose   |
+---------------+   +---------------+   +---------------+
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                 +--------------------+
                 | Driver-State Logic |
                 +--------------------+
                           |
                           v
                 +--------------------+
                 |  Visual / Audio    |
                 |      Warning       |
                 +--------------------+
```

---

## Features

- Real-time face detection using MediaPipe
- 98-point facial landmark detection
- Modified MobileNetV2 landmark regression model
- Grayscale face input for landmark prediction
- Eye Aspect Ratio (EAR) calculation
- Mouth Aspect Ratio (MAR) calculation
- Head pose estimation using OpenCV SolvePnP
- Pitch, yaw, and roll estimation
- Real-time webcam processing
- Landmark smoothing for real-time processing
- Driver-state warning logic
- Audio warning support
- Static image testing
- Model training using the WFLW dataset
- Model evaluation tools
- Google Colab training notebook

---

## Facial Landmark Model

The landmark detector is implemented using a customized **MobileNetV2** neural network.

### Input

```text
Grayscale face image
        |
        v
     112 x 112
```

### Output

The neural network predicts:

```text
98 landmarks x 2 coordinates = 196 outputs
```

Each landmark contains normalized:

```text
(x, y)
```

coordinates.

### Model Modifications

The original MobileNetV2 architecture is modified for the facial landmark regression task.

Main modifications include:

- Input changed from 3-channel RGB to 1-channel grayscale
- ReLU6 activations replaced with ReLU
- Additional convolutional processing
- Custom regression head
- Final output layer with 196 values

The regression head produces the coordinates of all 98 facial landmarks.

---

## Driver-State Analysis

### Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is calculated from facial landmarks around the eyes.

EAR provides geometric information about eye opening and can be used by the real-time monitoring logic to analyze eye state.

```text
98 Facial Landmarks
        |
        v
Eye Landmark Points
        |
        v
       EAR
```

---

### Mouth Aspect Ratio (MAR)

The Mouth Aspect Ratio is calculated using landmarks around the mouth.

MAR provides information about mouth opening and can be used as part of yawning-related analysis.

```text
98 Facial Landmarks
        |
        v
Mouth Landmark Points
        |
        v
       MAR
```

---

### Head Pose Estimation

Head orientation is estimated using OpenCV's **SolvePnP** method.

The system estimates:

- Pitch
- Yaw
- Roll

Pipeline:

```text
Facial Landmarks
        |
        v
Selected 2D Face Points
        |
        v
3D Face Reference Points
        |
        v
OpenCV SolvePnP
        |
        v
Pitch / Yaw / Roll
```

This information can be used by the monitoring logic to analyze the driver's head orientation.

---

## Project Structure

```text
DMS_Project1/
|
|-- assets/
|   |-- test.jpg
|   `-- output.jpg
|
|-- notebooks/
|   `-- Train_Colab.ipynb
|
|-- src/
|   |-- alarm_pip.wav
|   |-- dataset.py
|   |-- ear_utils.py
|   |-- evaluate.py
|   |-- Hinhnen.jpg
|   |-- model.py
|   |-- realtime.py
|   |-- test_image.py
|   `-- train.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- landmark.pth        # Download separately
```

### Source Files

| File | Description |
|---|---|
| `src/model.py` | Defines the modified MobileNetV2 facial landmark model |
| `src/dataset.py` | Loads and preprocesses the WFLW dataset |
| `src/train.py` | Trains the 98-point facial landmark model |
| `src/evaluate.py` | Evaluates the trained model |
| `src/ear_utils.py` | EAR, MAR, and head pose calculations |
| `src/test_image.py` | Runs landmark detection on a static image |
| `src/realtime.py` | Runs the real-time Driver Monitoring System |
| `notebooks/Train_Colab.ipynb` | Google Colab training workflow |

---

## Requirements

Recommended environment:

```text
Python 3.10
```

The main libraries used in this project are:

- PyTorch
- TorchVision
- OpenCV
- MediaPipe
- NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Driver-Monitoring-System.git
```

Move into the project directory:

```bash
cd AI-Driver-Monitoring-System
```

### 2. Create a virtual environment

Windows:

```bash
py -3.10 -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Important MediaPipe Compatibility Note

The current source code uses the legacy MediaPipe Solutions API:

```python
mp.solutions.face_detection
```

The tested project environment uses:

```text
mediapipe==0.10.21
numpy==1.26.4
opencv-contrib-python==4.11.0.86
```

Using incompatible newer versions of MediaPipe may cause:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

For reproducibility, install the versions specified in `requirements.txt`.

---

## Pretrained Model

The pretrained landmark model is:

```text
landmark.pth
```

The weight file is not stored directly in the Git repository.

After downloading the pretrained model, place it in the project root:

```text
AI-Driver-Monitoring-System/
|
|-- landmark.pth
|-- assets/
|-- notebooks/
|-- src/
|-- README.md
`-- requirements.txt
```

### Download

Download the pretrained 98-point facial landmark model from GitHub Releases:

[**Download landmark.pth (v1.0.0)**](https://github.com/datphan10/AI-Driver-Monitoring-System/releases/download/v1.0.0/landmark.pth)

After downloading, place `landmark.pth` in the project root:

```text
AI-Driver-Monitoring-System/
├── landmark.pth
├── assets/
├── notebooks/
├── src/
├── README.md
└── requirements.txt

---

## Run Static Image Test

The default test image is:

```text
assets/test.jpg
```

Run:

```bash
python src/test_image.py
```

The result is saved automatically to:

```text
assets/output.jpg
```

The output image contains:

- Detected face bounding box
- 98 predicted facial landmarks

---

## Run Real-Time Driver Monitoring

Make sure a webcam is connected and accessible.

Run:

```bash
python src/realtime.py
```

The real-time application processes camera frames and performs:

```text
Camera
  |
  v
Face Detection
  |
  v
98 Facial Landmarks
  |
  +--> EAR
  |
  +--> MAR
  |
  +--> Head Pose
  |
  v
Driver-State Analysis
  |
  v
Warning
```

---

## Training

The landmark model is trained using the **WFLW (Wider Facial Landmarks in-the-Wild)** dataset.

The training pipeline includes:

- Face image loading
- Bounding-box cropping
- Grayscale conversion
- Image resizing to 112 x 112
- Landmark normalization
- Wing Loss
- Increased weighting for eye landmarks
- Increased weighting for mouth landmarks
- Adam optimizer
- Cosine annealing learning-rate scheduler
- Gradient clipping
- Early stopping

Run training with:

```bash
python src/train.py
```

A Google Colab training workflow is also included:

```text
notebooks/Train_Colab.ipynb
```

This can be used to train the model with GPU acceleration in Google Colab.

---

## Training Configuration

The current training configuration includes:

| Parameter | Value |
|---|---:|
| Input size | 112 x 112 |
| Input channels | 1 |
| Facial landmarks | 98 |
| Model outputs | 196 |
| Batch size | 64 |
| Maximum epochs | 80 |
| Initial learning rate | 0.001 |
| Weight decay | 1e-5 |
| Optimizer | Adam |
| Loss function | Wing Loss |
| LR scheduler | Cosine Annealing |
| Early stopping patience | 10 |

Additional landmark weighting is applied during training:

```text
Eye landmarks   -> 5x weight
Mouth landmarks -> 2x weight
```

---

## Evaluation

The project includes:

```text
src/evaluate.py
```

for evaluating the trained landmark model.

Run:

```bash
python src/evaluate.py
```

Evaluation code can be used to analyze landmark prediction performance and inference behavior on the WFLW dataset.

---

## Technologies

```text
Python
PyTorch
TorchVision
OpenCV
MediaPipe
NumPy
WFLW Dataset
Google Colab
Computer Vision
Deep Learning
```

---

## Current Development Status

Implemented:

- [x] WFLW dataset loader
- [x] 98-point facial landmark model
- [x] Model training pipeline
- [x] Model evaluation
- [x] Static image inference
- [x] MediaPipe face detection
- [x] EAR calculation
- [x] MAR calculation
- [x] Head pose estimation
- [x] Real-time camera processing
- [x] Audio warning logic

Possible future improvements:

- [ ] Improve landmark accuracy under difficult lighting conditions
- [ ] Improve robustness for large head rotations
- [ ] Optimize inference performance for edge devices
- [ ] Export the landmark model to ONNX
- [ ] Apply model quantization
- [ ] Add more systematic real-time performance evaluation

---

## Disclaimer

This project is developed for educational and research purposes.

It is not intended to replace certified automotive driver-monitoring or safety systems.

---

## Author

Developed as a computer vision and Edge AI project.
