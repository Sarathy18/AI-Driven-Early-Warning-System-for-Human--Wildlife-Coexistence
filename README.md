**AI Wildlife Detection and Threat Alert System**
Overview
This project integrates computer vision, audio signal processing, and embedded systems to detect wildlife threats in real-time. By combining YOLO-based object detection, MFCC-CNN audio classification, and an ESP32 microcontroller, the system identifies potential wildlife threats and sends timely alerts to users via Bluetooth.

**Features**
Visual Detection: Utilizes YOLO (You Only Look Once) for real-time object detection of wildlife.

# Audio Classification: Employs MFCC (Mel Frequency Cepstral Coefficients) features with a CNN model to classify animal sounds.

# Sensor Integration: Incorporates an ultrasonic sensor with ESP32 to measure the distance of detected wildlife.

# Alert System: Sends structured alerts via Bluetooth, including time, location, animal type, and distance.

System Architecture
Data Acquisition:

Visual data captured using cameras.

Audio data recorded via microphones.

Distance measured using an ultrasonic sensor connected to ESP32.

### Processing:

YOLO processes visual data to detect and classify animals.

Audio data is transformed into MFCC features and classified using a CNN model.

ESP32 processes sensor data and coordinates alerts.

### Alert Mechanism:

Upon detection, ESP32 sends alerts via Bluetooth containing:

Timestamp

Location

Animal type

Approximate distance

YOLO Object Detection
YOLO (You Only Look Once) is a real-time object detection system that processes images in a single pass, making it highly efficient.

### Training Dataset: Custom dataset containing images of various wildlife species.

### Annotations: Labeled using YOLO format.

### Training Framework: Ultralytics YOLOv8 with PyTorch backend.

Inference
Real-time detection on video streams.

Outputs bounding boxes and class labels for detected animals.

###### MFCC-CNN Audio Classification
Audio classification is performed using MFCC features and a Convolutional Neural Network (CNN).

###### Feature Extraction
MFCC: Extracts key features from audio signals that represent the power spectrum of sounds.

Tools: Librosa library for feature extraction.

Model Architecture
Input: MFCC feature arrays.

Layers:

Convolutional layers for feature detection.

Pooling layers for dimensionality reduction.

Fully connected layers for classification.

Output: Probability distribution over animal classes.

Training & Evaluation
Dataset: Audio recordings of various animal sounds.

Training: Model trained to minimize categorical cross-entropy loss.

Evaluation Metrics: Accuracy, precision, recall, and F1-score.

ESP32 Integration
The ESP32 microcontroller handles sensor data processing and alert dissemination.

Hardware Components
Ultrasonic Sensor: Measures the distance to detected wildlife.

Buzzer & LED: Provide auditory and visual alerts.

Bluetooth Module: Sends alerts to connected devices.

Software Functionality
Reads distance measurements from the ultrasonic sensor.

Triggers buzzer and LED based on detected threats.

Formats and sends alert messages via Bluetooth, including:

Time of detection

Location

Animal type

Approximate distance

Installation Prerequisites

**Python 3.8+

ESP32 Board

Ultrasonic Sensor

Microphone

Camera Module**

Python Dependencies
Install the required Python packages using the provided requirements.txt:


```python
pip install -r requirements.txt
```

requirements.txt

```python
numpy
opencv-python
torch
torchvision
torchaudio
librosa
matplotlib
ultralytics
pyserial
```

Train YOLO Model:

- Prepare and annotate your dataset.

- Train the model using Ultralytics YOLOv8 framework.

Train Audio Classification Model:

- Collect and preprocess audio data.

- Extract MFCC features using Librosa.

- Train the CNN model on extracted features.

Deploy Models:

- Load trained models onto the processing unit (e.g., Raspberry Pi).

- Integrate with ESP32 for sensor data processing.

Run the System:

- Start the camera and microphone for data acquisition.

- ESP32 processes sensor data and sends alerts upon detection.