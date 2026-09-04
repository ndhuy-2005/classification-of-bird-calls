# 🐦 Bird Sound Classification using Improved CRNN

A Deep Learning system for **bird species recognition from audio recordings**, built with a hybrid **CNN + Frequency Attention + Bidirectional GRU (CRNN)** architecture.

The project covers the complete pipeline from **audio annotation and preprocessing → Log-Mel Spectrogram extraction → model training → evaluation → desktop application deployment**.

> 🎓 Course Project — Speech Processing
> 🏫 Ho Chi Minh City University of Technology and Education
> 📅 2025–2026

---

## 📌 Overview

Identifying bird species from audio recordings manually requires specialized knowledge and can be difficult to scale.

This project develops an automated bird sound classification system capable of recognizing **22 bird species across 4 families** from audio recordings.

The system converts audio signals into **Log-Mel Spectrograms**, then uses an **Improved CRNN** architecture combining:

* CNN for time-frequency feature extraction
* Frequency Attention for emphasizing informative frequency bands
* Bidirectional GRU for temporal modeling
* Fully Connected layer for multi-class classification

A desktop GUI is also provided, allowing users to upload an audio file and receive the predicted bird species and confidence score.

The dataset contains approximately:

* **3,500 training samples**
* **750 validation samples**
* **750 test samples**

with recordings sourced from **Xeno-canto**.

---

## ✨ Key Features

### 🎵 Audio Processing

* Load audio at **48 kHz, mono**
* Band-pass filtering in the **1–8 kHz** frequency range
* Automatic detection of the most relevant 3-second bird sound segment
* Log-Mel Spectrogram extraction
* Fixed temporal dimension of **282 frames**
* Spectrogram normalization and clipping

### 🧠 Deep Learning Model

**ImprovedCRNN**

```text
Input
(1 × 128 × 282)
       │
       ▼
┌─────────────────┐
│   CNN Backbone  │
│  1 → 64 → 128   │
│  → 256 → 256    │
└────────┬────────┘
         │
         ▼
Frequency Attention
         │
         ▼
Weighted Frequency Aggregation
         │
         ▼
Bidirectional GRU
      2 Layers
         │
         ▼
      Dropout
         │
         ▼
Fully Connected
      22 Classes
```

The CNN backbone consists of four convolutional blocks using:

* Conv2D
* Batch Normalization
* ReLU
* Max Pooling

The resulting features are passed through a **Frequency Attention** mechanism before temporal modeling with a **2-layer Bidirectional GRU**.

---

## 🔬 Data Augmentation

To improve generalization and reduce overfitting, the training pipeline uses multiple augmentation techniques:

* **SpecAugment**

  * Frequency Masking: 30 bins
  * Time Masking: 70 frames
* Gaussian Noise
* Gain Adjustment: 0.6–1.5
* Time Shift: ±80 frames
* Frequency Shift: ±15 bins
* **Mixup**

  * Probability: 60%
  * Beta distribution: α = 0.4

Class imbalance is addressed using **class-weighted loss**.

---

## 🗂️ Data Pipeline

The annotation pipeline starts from **Sonic Visualiser Layer (.svl)** files.

```text
SVL Annotation
      │
      ▼
Parse XML
      │
      ▼
Normalize Class Labels
      │
      ▼
CSV Annotation
      │
      ▼
Extract Audio Segments
      │
      ▼
Log-Mel Spectrogram
      │
      ▼
.npy Spectrogram Dataset
      │
      ▼
PyTorch Dataset / DataLoader
      │
      ▼
Model Training
```

The original annotations contain frame, duration, frequency information and bird species labels. They are parsed into:

```text
train_annotation.csv
val_annotation.csv
test_annotation.csv
```

## Spectrograms are then stored as `.npy` files under separate `train`, `val`, and `test` directories.

## 🐦 Supported Species

The dataset contains **22 species** from four families:

| Family        | Number of Species |
| ------------- | ----------------: |
| Fringillidae  |                 2 |
| Paridae       |                 9 |
| Troglodytidae |                 4 |
| Turdidae      |                 7 |
| **Total**     |            **22** |

---

## ⚙️ Training Configuration

| Parameter             | Value             |
| --------------------- | ----------------- |
| Framework             | PyTorch           |
| Batch Size            | 32                |
| Maximum Epochs        | 100               |
| Optimizer             | AdamW             |
| Scheduler             | ReduceLROnPlateau |
| Initial Learning Rate | 5 × 10⁻⁴          |
| Weight Decay          | 1 × 10⁻⁴          |
| Loss                  | Cross Entropy     |
| Label Smoothing       | 0.1               |
| Early Stopping        | Patience = 30     |
| Mixup Probability     | 60%               |

The best model is selected according to **Validation Weighted F1-score** and saved as `best_model.pth`.

---

## 📊 Results

The best model was obtained at **epoch 44**:

| Metric            |      Score |
| ----------------- | ---------: |
| Validation Loss   |  **1.654** |
| Weighted F1-score | **0.6740** |
| Macro F1-score    | **0.6589** |

The model was also compared against several baselines:

| Method                        |  F1-score |
| ----------------------------- | --------: |
| **Improved CRNN + Attention** | **0.674** |
| ResNet-50                     |     0.651 |
| LSTM Baseline                 |     0.589 |
| Traditional ML — SVM          |     0.523 |

The results indicate that combining convolutional, attention-based and recurrent components provides better performance for this classification task than the evaluated baselines.

---

## 🖥️ Desktop Application

The trained model is deployed as a desktop application using **Tkinter**.

### Application workflow

```text
Select Audio File
       │
       ▼
Load Audio
       │
       ▼
Detect Bird Segment
       │
       ▼
Log-Mel Spectrogram
       │
       ▼
Normalize
       │
       ▼
ImprovedCRNN
       │
       ▼
Softmax
       │
       ▼
Bird Species + Confidence
```

Supported audio formats:

```text
.wav
.mp3
.flac
.ogg
.m4a
```

The application displays:

* Predicted bird species
* Prediction confidence
* Bird image
* Processing status

The application can automatically select the highest-energy **3-second segment** from longer recordings before classification.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scipy pillow
```

> Make sure the installed PyTorch version is compatible with your Python environment.

---

## ▶️ Run the Application

Place the trained model at:

```text
best_model.pth
```

Then run:

```bash
python bird_classifier_gui.py
```

The application automatically selects CUDA when available and falls back to CPU otherwise.

---

## 📁 Project Structure

```text
bird-sound-classification/
│
├── notebooks/
│   ├── xltn-to-csv.ipynb
│   ├── xltn-audit.ipynb
│   └── xltn-train.ipynb
│
├── spectrograms/
│   ├── train/
│   ├── val/
│   └── test/
│
├── class_images/
│   └── *.jpg
│
├── train_annotation.csv
├── val_annotation.csv
├── test_annotation.csv
│
├── best_model.pth
├── bird_classifier_gui.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

**Programming**

* Python

**Deep Learning**

* PyTorch
* CNN
* GRU
* Frequency Attention

**Audio Processing**

* Librosa
* SciPy
* NumPy
* Log-Mel Spectrogram

**Data Processing**

* XML / SVL
* CSV
* NumPy `.npy`

**Application**

* Tkinter
* Pillow

---

## ⚠️ Limitations

The current system has several limitations:

* Classification is limited to **22 species**
* Dataset contains class imbalance
* Performance decreases on noisy real-world recordings
* No real-time microphone recording
* No batch processing
* No prediction history

On unseen real-world recordings, the reported accuracy is approximately **62–65%**, lower than validation performance due to recording quality, environmental noise and variation in bird vocalizations.

---

## 🔮 Future Improvements

### Short Term

* Expand the dataset to **100+ bird species**
* Improve class balance
* Experiment with EfficientNet and Transformer architectures
* Apply transfer learning
* Ensemble multiple models
* Add real-time recording
* Add spectrogram visualization
* Support batch prediction

### Long Term

* Deploy as a Web API using Flask/FastAPI
* Develop mobile applications
* Cloud deployment
* Multi-species detection
* Temporal bird activity analysis
* Geolocation integration
* Few-shot and self-supervised learning
* Domain adaptation

These directions are intended to improve robustness and extend the system toward real-world ecological monitoring applications.

---

## 🌱 Potential Applications

The system can potentially support:

* Ecological monitoring
* Bird population monitoring
* Biodiversity assessment
* Migration tracking
* Nature conservation
* Rare species detection
* Educational applications
* Citizen science projects

---

## 📚 References

1. Xeno-canto Foundation — Bird sound database
2. He et al. — Deep Residual Learning for Image Recognition
3. Park et al. — SpecAugment
4. Zhang et al. — Mixup
5. McFee et al. — Librosa
6. Paszke et al. — PyTorch
7. Kahl et al. — Large-Scale Bird Sound Classification
8. Pellegrini — Densely Connected CNNs for Bird Audio Detection
9. Salamon & Bello — Environmental Sound Classification
10. Çakır et al. — Convolutional Recurrent Neural Networks for Sound Event Detection

---

## 👥 Team

**IT3 — Speech Processing Project**

* Nguyễn Đình Huy
* Lục Thế Vỹ
* Đặng Thiên Bách
* Dương Minh Tâm

---

## 📄 License

This project was developed for academic purposes.
