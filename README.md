# 🩺 Skin Disease Classification System
> Deep learning-based dermoscopic image classifier using EfficientNetB3  
> Trained on HAM10000 · 7 skin conditions · Flask web app included

---

## 📁 Project Structure

```
skin_disease_classifier/
├── src/
│   ├── train.py          # Model training (EfficientNetB3, two-phase)
│   ├── predict.py        # Inference module (single + batch)
│   └── app.py            # Flask REST API + web server
├── utils/
│   └── data_utils.py     # EDA, preprocessing, dataset balancing
├── templates/
│   └── index.html        # Web UI (dark theme, drag-and-drop)
├── notebooks/
│   └── skin_disease_notebook.ipynb  # Jupyter experimentation
├── models/               # Saved model weights (created at training time)
├── data/                 # Dataset directory (see download steps)
├── uploads/              # Temp upload storage (created automatically)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the HAM10000 Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key: https://www.kaggle.com/settings → API → Create Token
mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download kmader/skin-cancer-mnist-ham10000 -p data/
unzip data/skin-cancer-mnist-ham10000.zip -d data/
```

Expected `data/` layout:
```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/   (5000 images)
  HAM10000_images_part_2/   (5015 images)
```

### 3. Run Exploratory Data Analysis

```bash
python utils/data_utils.py --eda --csv data/HAM10000_metadata.csv --image_dir data/
# Output plots saved to: eda_output/
```

### 4. Train the Model

```bash
python src/train.py --csv data/HAM10000_metadata.csv --image_dir data/
# Trained model saved to: models/skin_disease_final.h5
# Training curves and confusion matrix saved to: models/
```

> 💡 **Google Colab** is recommended for GPU training. Open `notebooks/skin_disease_notebook.ipynb` and mount your Drive.

### 5. Launch the Web Application

```bash
python src/app.py
# Open: http://localhost:5000
```

> Without a trained model, the app runs in **demo mode** with simulated predictions for UI testing.

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | EfficientNetB3 (ImageNet pretrained) |
| Input size | 224 × 224 × 3 |
| Head | GAP → BN → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax(7) |
| Training | Two-phase: head-only (20 ep) + fine-tune top 30 layers (30 ep) |
| Optimizer | Adam (phase 1: 1e-4, phase 2: 1e-5) |
| Loss | Categorical crossentropy with class weighting |
| Augmentation | Rotation, flips, zoom, shift, brightness |

---

## 🏷️ Supported Classes

| Code | Class Name | Risk Level |
|------|-----------|------------|
| nv   | Melanocytic Nevi | Low |
| mel  | Melanoma | **High** |
| bkl  | Benign Keratosis | Low |
| bcc  | Basal Cell Carcinoma | Medium-High |
| akiec | Actinic Keratosis | Medium |
| vasc | Vascular Lesions | Low |
| df   | Dermatofibroma | Low |

---

## 📊 REST API

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Web UI |
| `/api/predict` | POST | Upload image, get prediction |
| `/api/classes` | GET | List all classes + metadata |
| `/api/health` | GET | Health check |

**Example request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@path/to/skin_image.jpg"
```

**Example response:**
```json
{
  "success": true,
  "predicted_class": "Melanoma",
  "confidence": 91.4,
  "risk": "High",
  "description": "Melanoma is the most dangerous form of skin cancer...",
  "action": "URGENT: Seek immediate dermatological consultation.",
  "top_k": [
    {"class": "Melanoma", "confidence": 91.4, "risk": "High"},
    {"class": "Melanocytic Nevi", "confidence": 5.2, "risk": "Low"},
    {"class": "Benign Keratosis", "confidence": 2.1, "risk": "Low"}
  ],
  "all_probabilities": { ... }
}
```

---

## 📈 Expected Performance (HAM10000)

| Metric | Expected Value |
|--------|---------------|
| Accuracy | ~85–90% |
| AUC | ~0.95+ |
| Macro F1 | ~0.80+ |

*Actual results depend on hardware, epochs, and augmentation settings.*

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
It is **not** a certified medical device and must **not** be used for clinical diagnosis.  
Always consult a qualified dermatologist for any skin condition concerns.

---

## 📚 References

- **Dataset**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Paper**: Tschandl et al. (2018) — *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*
- **Model**: [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
