# 🚗 Driver Drowsiness Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A real-time hybrid machine learning system for detecting driver drowsiness using computer vision and deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [ML Libraries](#-machine-learning-libraries)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Machine Learning Libraries](#-machine-learning-libraries)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This project implements a **hybrid driver drowsiness detection system** that combines multiple machine learning approaches to accurately identify signs of drowsiness in real-time. The system uses:

- **Eye Aspect Ratio (EAR)** analysis with traditional ML classifiers
- **Convolutional Neural Networks (CNNs)** for eye state classification
- **Deep learning models** for yawn detection
- **Facial landmark detection** for precise feature extraction

The system processes live webcam feed and triggers alerts when drowsiness indicators are detected, helping prevent accidents caused by driver fatigue.

---

## ✨ Features

- 🔍 **Multi-Modal Detection**: Combines EAR-based ML, CNN eye classification, and yawn detection
- ⚡ **Real-Time Processing**: Live webcam feed analysis with minimal latency
- 🎯 **Facial Landmark Detection**: Precise 68-point facial landmark extraction using dlib
- 📊 **Visual Feedback**: Real-time metrics display (EAR, eye probability, yawn probability)
- 🚨 **Smart Alerts**: Context-aware alerts for different drowsiness indicators
- 📝 **Debug Logging**: Automatic logging of alerts and debug images
- 🔄 **Robust Training Pipeline**: Separate training scripts for each model component

---

## 🚀 Installation

### Prerequisites

- Python 3.11
- Windows 10/11 (or compatible OS)
- Webcam/Camera access
- Visual C++ Build Tools (for dlib on Windows)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Driver-Drowsiness-Detection
```

### Step 2: Create & Activate Virtual Environment

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Step 4: Download dlib Shape Predictor

Download the required facial landmark predictor file:

```bash
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in project root as: shape_predictor_68_face_landmarks.dat
```

---

## 💻 Usage

### Running the Detector

After ensuring all models are trained and available in the `models/` directory:

```bash
python src\drowsiness_detector.py
```

**Controls:**
- Press `q` to quit the application
- The system will display real-time metrics and alerts on the video feed

### Training Models

```bash
# Train Eye CNN
python src\eye_cnn_training.py --dataset dataset --out models\cnn_model.h5 --epochs 15

# Train Mouth/Yawn CNN
python src\mouth_cnn_training.py

# Train EAR-based ML Classifier
python src\ml_classifier_training.py --dataset dataset --out models\ml_model.pkl
```

## 📁 Project Structure

```
Driver-Drowsiness-Detection/
│
├── 📂 src/                          # Main source code
│   ├── drowsiness_detector.py      # Real-time detection application
│   ├── eye_cnn_training.py         # Eye CNN training script
│   ├── mouth_cnn_training.py       # Mouth/Yawn CNN training script
│   └── ml_classifier_training.py   # EAR-based ML classifier training
│
├── 📂 utils/                        # Utility modules
│   ├── eye_aspect_ratio.py         # EAR calculation implementation
│   └── landmark_utils.py           # Facial landmark extraction helpers
│
├── 📂 scripts/                      # Helper scripts
│   ├── detector_limited_run.py     # Limited frame test runner
│   ├── detector_smoke_test.py      # System smoke test
│   ├── smoke_test_imports.py       # Import verification
│   └── run_all_training.cmd        # Batch training script
│
├── 📂 models/                       # Trained model files
│   ├── cnn_model.h5                # Eye open/closed CNN model
│   ├── mouth_cnn_model.h5          # Yawn detection CNN model
│   └── ml_model.pkl                # EAR-based RandomForest classifier
│
├── 📂 dataset/                      # Training dataset
│   ├── open_eyes/                  # Open eye images (~726 images)
│   ├── closed_eyes/                # Closed eye images (~726 images)
│   ├── yawn/                       # Yawning images (~723 images)
│   └── no_yawn/                    # Non-yawning images (~725 images)
│
├── 📂 outputs/                      # Generated outputs
│   ├── accuracy_plot.png           # Training accuracy curves
│   ├── loss_plot.png               # Training loss curves
│   ├── alerts.log                  # Alert event log
│   └── debug_mouth/                # Debug mouth crop images
│
├── 📂 drowzy/                       # Python virtual environment
├── shape_predictor_68_face_landmarks.dat  # dlib facial landmark predictor
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📄 File Descriptions

### Source Files (`src/`)

#### `drowsiness_detector.py`
**Purpose**: Main real-time detection application

**Functionality**:
- Opens webcam feed using OpenCV
- Detects faces using dlib's frontal face detector
- Extracts 68-point facial landmarks
- Computes Eye Aspect Ratio (EAR) for both eyes
- Runs three parallel detection models:
  - EAR-based RandomForest classifier
  - Eye CNN for open/closed classification
  - Mouth CNN for yawn detection
- Monitors for sideways head position
- Triggers alerts when drowsiness indicators persist
- Displays real-time metrics and visual feedback
- Logs alerts and saves debug images

**Key Features**:
- Multi-threshold detection system
- Consecutive frame counting for stability
- Context-aware alert messages
- Debug image saving for analysis

---

#### `eye_cnn_training.py`
**Purpose**: Train CNN model for eye open/closed classification

**Functionality**:
- Builds a sequential CNN with Conv2D, MaxPooling2D, and Dense layers
- Uses TensorFlow/Keras ImageDataGenerator for data augmentation
- Trains on `dataset/open_eyes/` and `dataset/closed_eyes/` directories
- Implements validation split (15%) for model evaluation
- Applies data augmentation (rotation, horizontal flip)
- Saves best model to `models/cnn_model.h5`
- Generates accuracy and loss plots
- Includes early stopping and model checkpointing

**Model Architecture**:
- Input: 64x64x3 RGB images
- Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Flatten → Dense(128) → Dense(1, sigmoid)
- Binary classification with alphabetical directory labels: closed_eyes=0, open_eyes=1

---

#### `mouth_cnn_training.py`
**Purpose**: Train CNN model for yawn detection

**Functionality**:
- Builds a deeper CNN architecture for mouth/yawn classification
- Processes grayscale images (64x64x1)
- Trains on `dataset/yawn/` and `dataset/no_yawn/` directories
- Uses extensive data augmentation (rotation, shifts, brightness, zoom)
- Implements dropout (0.5) for regularization
- Saves model to `models/mouth_cnn_model.h5`

**Model Architecture**:
- Input: 64x64x1 grayscale images
- Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Conv2D(128) → MaxPooling2D → Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
- Binary classification (no_yawn=0, yawn=1)

---

#### `ml_classifier_training.py`
**Purpose**: Train traditional ML classifier based on Eye Aspect Ratio features

**Functionality**:
- Extracts EAR features from training images using dlib landmarks
- Computes left EAR, right EAR, and average EAR for each image
- Trains a RandomForest classifier (100 estimators)
- Handles face detection failures with fallback to OpenCV Haar cascades
- Splits data into train/test sets (80/20)
- Evaluates model with accuracy and classification report
- Saves trained model to `models/ml_model.pkl` using pickle

**Features**:
- Robust face detection with multiple fallback strategies
- Comprehensive logging of training progress
- Path resolution for cross-platform compatibility

---

### Utility Files (`utils/`)

#### `eye_aspect_ratio.py`
**Purpose**: Calculate Eye Aspect Ratio from facial landmarks

**Functionality**:
- Implements the EAR formula: `(||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
- Takes 6 eye landmark points as input
- Returns EAR value (lower = more closed, higher = more open)
- Handles edge cases (zero division)

**Mathematical Formula**:
```
EAR = (vertical_distances_sum) / (2 * horizontal_distance)
```

---

#### `landmark_utils.py`
**Purpose**: Helper functions for facial landmark processing

**Functions**:
- `shape_to_coords(shape)`: Converts dlib shape object to list of (x,y) coordinates
- `get_left_eye(coords)`: Extracts left eye points (indices 42-47)
- `get_right_eye(coords)`: Extracts right eye points (indices 36-41)
- `crop_eye(image, eye_points, margin, size)`: Crops eye/mouth region from image with padding

**Usage**: Used throughout the project for extracting specific facial regions for CNN processing

---

## 🤖 Machine Learning Libraries

### Core Deep Learning

#### **TensorFlow (2.12.0)**
**Purpose**: Deep learning framework for building and training neural networks

**Usage in Project**:
- **Keras API**: Used to build CNN architectures for eye and mouth classification
- **Model Building**: Sequential models with Conv2D, MaxPooling2D, Dense layers
- **Model Training**: `model.fit()` with ImageDataGenerator for data augmentation
- **Model Loading**: `load_model()` for real-time inference in detector
- **Callbacks**: ModelCheckpoint and EarlyStopping for training optimization

**Key Features Used**:
- Image preprocessing and normalization
- Data augmentation (rotation, flipping, brightness adjustment)
- Binary classification with sigmoid activation
- Model serialization (HDF5 format)

---

#### **h5py (3.15.1)**
**Purpose**: Python interface to HDF5 binary data format

**Usage in Project**:
- **Model Storage**: Saves and loads Keras/TensorFlow models in `.h5` format
- **Efficient I/O**: Fast reading/writing of large model weights
- **Model Persistence**: Enables model reuse across training and inference sessions

**Files**: `models/cnn_model.h5`, `models/mouth_cnn_model.h5`

---

### Computer Vision

#### **OpenCV-Python (4.7.0.72)**
**Purpose**: Computer vision and image processing library

**Usage in Project**:
- **Video Capture**: `cv2.VideoCapture(0)` for webcam feed access
- **Image Processing**: Color conversion (`COLOR_BGR2GRAY`), resizing, cropping
- **Face Detection**: Haar cascade fallback when dlib fails
- **Visualization**: Real-time video display with text overlays and bounding boxes
- **Image I/O**: Reading training images, saving debug crops
- **Preprocessing**: Image normalization and resizing for CNN input

**Key Functions Used**:
- `cv2.imread()`, `cv2.imwrite()` - Image I/O
- `cv2.cvtColor()` - Color space conversion
- `cv2.resize()` - Image resizing
- `cv2.putText()`, `cv2.rectangle()` - Drawing overlays
- `cv2.imshow()`, `cv2.waitKey()` - Video display

---

#### **dlib (20.0.0)**
**Purpose**: Machine learning and computer vision library, specifically for facial landmark detection

**Usage in Project**:
- **Face Detection**: `dlib.get_frontal_face_detector()` for detecting faces in images
- **Facial Landmarks**: `dlib.shape_predictor()` extracts 68-point facial landmarks
- **Feature Extraction**: Provides precise coordinates for eyes, nose, mouth regions
- **EAR Calculation**: Landmark points used to compute Eye Aspect Ratio

**Key Components**:
- **Frontal Face Detector**: HOG-based face detection
- **Shape Predictor**: Pre-trained model for 68-point landmark detection
- **Required File**: `shape_predictor_68_face_landmarks.dat` (must be downloaded separately)

**Landmark Indices Used**:
- Left Eye: 36-41
- Right Eye: 42-47
- Nose Tip: 30
- Mouth: 48-67

---

### Traditional Machine Learning

#### **scikit-learn (1.3.2)**
**Purpose**: Machine learning library for traditional ML algorithms

**Usage in Project**:
- **RandomForestClassifier**: Trains EAR-based drowsiness classifier
- **Model Evaluation**: `accuracy_score()`, `classification_report()`
- **Data Splitting**: `train_test_split()` for validation
- **Model Persistence**: Used with pickle for model serialization

**Key Features**:
- Ensemble learning with 100 decision trees
- Feature-based classification (EAR values)
- Fast inference for real-time applications

---

#### **NumPy (1.23.5)**
**Purpose**: Fundamental package for numerical computing in Python

**Usage in Project**:
- **Array Operations**: Image data manipulation, coordinate arrays
- **Mathematical Operations**: EAR calculations, distance computations
- **Data Preprocessing**: Array reshaping, normalization, type conversion
- **CNN Input**: Converting images to numpy arrays for TensorFlow
- **Coordinate Handling**: Processing facial landmark coordinates

**Key Operations**:
- Array creation and manipulation
- Mathematical functions (hypot for distance)
- Type conversions (float32 for CNN input)
- Array indexing and slicing

---

### Scientific Computing

#### **SciPy (1.15.3)**
**Purpose**: Scientific computing library built on NumPy

**Usage in Project**:
- **Dependency**: Required by scikit-learn and other ML libraries
- **Numerical Routines**: Used indirectly for statistical operations
- **Image Processing**: Advanced image operations if needed

---

### Visualization & Plotting

#### **Matplotlib (3.10.7)**
**Purpose**: Plotting and visualization library

**Usage in Project**:
- **Training Curves**: Generates accuracy and loss plots during training
- **Model Evaluation**: Visual representation of training progress
- **Output Files**: Saves plots to `outputs/accuracy_plot.png` and `outputs/loss_plot.png`

**Key Features Used**:
- Line plots for training history
- Legend and labeling
- Figure saving

---

### Image Processing

#### **Pillow (12.0.0)**
**Purpose**: Python Imaging Library (PIL) for image manipulation

**Usage in Project**:
- **Image Utilities**: Additional image processing capabilities
- **Format Support**: Handling various image formats
- **Backup Processing**: Alternative image operations if needed

---

### Web Framework (Optional)

#### **FastAPI (0.121.2)**
**Purpose**: Modern web framework for building APIs

**Usage in Project**:
- **Optional Feature**: Can be used to create web API wrapper for the detector
- **Remote Access**: Enables browser-based or remote access to detector
- **RESTful API**: REST endpoints for detector functionality

---

#### **Uvicorn (0.38.0)**
**Purpose**: ASGI web server implementation

**Usage in Project**:
- **Server**: Runs FastAPI applications
- **Development**: Local server for testing web API
- **Production**: Can be used for deployment

---

### Build Tools

#### **CMake (4.1.2)**
**Purpose**: Cross-platform build system

**Usage in Project**:
- **dlib Compilation**: Required for building dlib from source on Windows
- **Native Extensions**: Builds C++ extensions for Python packages
- **Dependency**: Needed if installing dlib without prebuilt wheels

---

### Utilities

#### **requests (2.32.5)**
**Purpose**: HTTP library for making web requests

**Usage in Project**:
- **Resource Download**: Can be used to download model files or datasets
- **API Integration**: Potential for remote model serving or data fetching

---

## 🔄 How It Works

### Detection Pipeline

1. **Face Detection**: dlib detects faces in the video frame
2. **Landmark Extraction**: 68-point facial landmarks are extracted
3. **Feature Extraction**:
   - Eye landmarks → EAR calculation
   - Eye region → CNN preprocessing
   - Mouth region → CNN preprocessing
4. **Multi-Model Inference**:
   - EAR → RandomForest classifier
   - Eye region → Eye CNN
   - Mouth region → Mouth CNN
5. **Decision Fusion**: Alerts triggered when indicators persist across consecutive frames
6. **Visualization**: Real-time display with metrics and alerts

### Alert Triggers

- **EAR Alert**: EAR < 0.23 for 20+ consecutive frames
- **Eye CNN Alert**: Eye closed probability ≥ 0.5 for 5+ consecutive frames
- **Yawn Alert**: Yawn probability ≥ 0.5 for 3+ consecutive frames
- **Sideways Alert**: Head turned left/right for 3+ seconds

---

## 🛠️ Troubleshooting

### Common Issues

#### dlib Installation Failures
**Problem**: `dlib` fails to install on Windows

**Solutions**:
1. Install Visual C++ Build Tools (C++ workload)
2. Install CMake: `pip install cmake`
3. Use prebuilt wheel from Christoph Gohlke's repository
4. Use conda: `conda install -c conda-forge dlib`

#### TensorFlow/NumPy Compatibility
**Problem**: `numpy>=1.24` incompatible with TensorFlow 2.12

**Solution**: Project pins `numpy==1.23.5` in `requirements.txt` to avoid conflicts

#### Missing Shape Predictor
**Problem**: `FileNotFoundError` for `shape_predictor_68_face_landmarks.dat`

**Solution**: Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract to project root

#### Camera Access Issues
**Problem**: Camera not opening or no frames captured

**Solutions**:
1. Check camera permissions
2. Verify camera index (try 0, 1, 2)
3. Ensure no other application is using the camera
4. Test with `scripts/detector_smoke_test.py`

#### Model Loading Errors
**Problem**: Models fail to load or are missing

**Solution**: Train models first using `scripts/run_all_training.cmd` or individual training scripts

---

## 📊 Dataset Information

The project expects a dataset organized as follows:

- **open_eyes/**: ~726 images of open eyes
- **closed_eyes/**: ~726 images of closed eyes
- **yawn/**: ~723 images of yawning faces
- **no_yawn/**: ~725 images of non-yawning faces

**Total**: ~2,900 training images

---

## 🎓 Model Performance

### Training Metrics
- Models are saved with best validation performance
- Early stopping prevents overfitting
- Data augmentation improves generalization
- Validation split ensures unbiased evaluation

### Real-Time Performance
- Processes frames at ~15-30 FPS (depending on hardware)
- Low latency alert system
- Efficient model inference

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

<div align="center">

**Built with ❤️ using Python, TensorFlow, OpenCV, and dlib**

⭐ Star this repo if you find it helpful!

</div>


---

## Streamlit Web Deployment

The repository includes a browser-based Streamlit interface in app.py. It uses WebRTC to receive webcam frames from the visitor's browser; it does not try to open a camera attached to the remote Streamlit server. The original desktop application remains available through:

```bash
python src/drowsiness_detector.py
```

### Streamlit Community Cloud settings

| Setting | Value |
|---|---|
| Repository | AbhiDevLab/driver-drowsiness-detection |
| Branch | master |
| Main file path | app.py |
| Python version | 3.11 |

### Required trained assets

The current Git repository intentionally excludes model binaries. The web detector needs all four files:

- cnn_model.h5 — trained eye-state CNN
- mouth_cnn_model.h5 — trained yawn CNN
- ml_model.pkl — trained EAR Random Forest classifier
- shape_predictor_68_face_landmarks.dat — dlib 68-point facial-landmark predictor

For local use, put the three model files inside models/ and the dlib predictor in the repository root. The app detects those files automatically.

For Streamlit Community Cloud, host the files somewhere that provides direct HTTPS downloads. Then open App → Settings → Secrets and add:

```toml
EYE_CNN_URL = "https://your-host/cnn_model.h5"
MOUTH_CNN_URL = "https://your-host/mouth_cnn_model.h5"
EAR_MODEL_URL = "https://your-host/ml_model.pkl"
DLIB_PREDICTOR_URL = "https://your-host/shape_predictor_68_face_landmarks.dat"
EYE_CNN_POSITIVE_CLASS = "open"
```

Set EYE_CNN_POSITIVE_CLASS to "open" when the eye CNN's sigmoid output represents the open_eyes class, or "closed" when it represents closed_eyes. The default "open" matches alphabetical class ordering when the training directory contains only closed_eyes and open_eyes. Check the class_indices printed by the original training generator before choosing; the H5 model does not preserve that mapping.

Use the extracted .dat predictor file, not the compressed .bz2 archive. These URLs are read as secrets and downloaded to an ignored local cache at startup. Do not commit private download tokens or .streamlit/secrets.toml to Git.

Optional TURN server settings can be added if the browser cannot establish a WebRTC connection through the default public STUN server:

```toml
TURN_SERVER_URL = "turn:your-turn-server:3478"
TURN_SERVER_USERNAME = "your-username"
TURN_SERVER_CREDENTIAL = "your-password"
```

### Local run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local localhost URL, allow camera access, and start the stream. On Streamlit Community Cloud, HTTPS is provided by the hosting platform. Some restrictive networks may still require TURN configuration.

### Important limitations

- Camera frames are transmitted from the browser to the hosted server for inference. The app does not intentionally save or record the video.
- All four trained assets must be supplied before live detection can start. A deployment without them displays an asset-configuration guide rather than a false working demo.
- The hosted application is an educational prototype, not a medically validated or vehicle-certified driver-safety product.
