"""
Skin Disease Classification - Inference / Prediction Module
"""

import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224

CLASS_NAMES = [
    'Melanocytic Nevi',
    'Melanoma',
    'Benign Keratosis',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Vascular Lesions',
    'Dermatofibroma',
]

# Clinical descriptions for each class
CLASS_INFO = {
    'Melanocytic Nevi': {
        'short': 'Common mole (benign)',
        'description': 'Melanocytic nevi are common benign skin lesions made up of melanocytes. '
                       'They are usually harmless but should be monitored for changes in shape, '
                       'size, or color.',
        'risk': 'Low',
        'action': 'Routine monitoring. Consult a dermatologist if changes occur.',
        'color': '#22c55e',
    },
    'Melanoma': {
        'short': 'Skin cancer (malignant)',
        'description': 'Melanoma is the most dangerous form of skin cancer, arising from melanocytes. '
                       'Early detection is critical for successful treatment.',
        'risk': 'High',
        'action': 'URGENT: Seek immediate dermatological consultation.',
        'color': '#ef4444',
    },
    'Benign Keratosis': {
        'short': 'Non-cancerous skin growth',
        'description': 'Benign keratosis-like lesions include seborrheic keratoses, solar lentigines, '
                       'and lichen planus-like keratoses. These are non-cancerous growths.',
        'risk': 'Low',
        'action': 'Generally no treatment needed. Cosmetic removal available.',
        'color': '#22c55e',
    },
    'Basal Cell Carcinoma': {
        'short': 'Skin cancer (slow-growing)',
        'description': 'Basal Cell Carcinoma is the most common type of skin cancer. It grows slowly '
                       'and rarely spreads, but requires treatment to prevent local tissue damage.',
        'risk': 'Medium-High',
        'action': 'Consult a dermatologist for treatment planning.',
        'color': '#f97316',
    },
    'Actinic Keratosis': {
        'short': 'Pre-cancerous lesion',
        'description': 'Actinic keratoses are rough, scaly patches caused by years of UV exposure. '
                       'They are considered pre-cancerous and may progress to squamous cell carcinoma.',
        'risk': 'Medium',
        'action': 'Dermatologist evaluation recommended for treatment.',
        'color': '#eab308',
    },
    'Vascular Lesions': {
        'short': 'Blood vessel anomaly',
        'description': 'Vascular lesions include angiomas, angiokeratomas, and pyogenic granulomas. '
                       'They arise from blood vessels and are usually benign.',
        'risk': 'Low',
        'action': 'Generally benign. Consult for cosmetic removal if desired.',
        'color': '#22c55e',
    },
    'Dermatofibroma': {
        'short': 'Benign skin nodule',
        'description': 'Dermatofibromas are harmless fibrous nodules that commonly appear on the legs. '
                       'They are benign and rarely require treatment.',
        'risk': 'Low',
        'action': 'No treatment required unless symptomatic or cosmetically bothersome.',
        'color': '#22c55e',
    },
}


# ─── MODEL LOADER ────────────────────────────────────────────────────────────
_model = None

def load_model(model_path: str = None) -> keras.Model:
    """Load trained model (singleton pattern)."""
    global _model
    if _model is not None:
        return _model

    if model_path is None:
        # Try common locations
        candidates = [
            'models/skin_disease_final.h5',
            'models/best_model_phase2.h5',
            'models/best_model_phase1.h5',
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        _model = keras.models.load_model(model_path)
    else:
        print("No saved model found — using demo mode (random predictions).")
        _model = None

    return _model


# ─── IMAGE PREPROCESSING ─────────────────────────────────────────────────────
def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess image for model inference.
    Accepts: file path (str), bytes, PIL Image, or numpy array.
    Returns: (1, IMG_SIZE, IMG_SIZE, 3) float32 array in [0, 1].
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── PREDICTION ──────────────────────────────────────────────────────────────
def predict(image_input, model_path: str = None, top_k: int = 3) -> dict:
    """
    Run inference on a single image.

    Returns:
        {
          'predicted_class': str,
          'confidence': float,
          'top_k': [{'class': str, 'confidence': float, 'info': dict}, ...],
          'all_probabilities': {class_name: float, ...},
          'info': dict,
        }
    """
    model = load_model(model_path)
    arr   = preprocess_image(image_input)

    if model is not None:
        probs = model.predict(arr, verbose=0)[0]
    else:
        # Demo: softmax-like random for testing without a trained model
        raw   = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
        probs = raw

    idx_sorted   = np.argsort(probs)[::-1]
    top_class    = CLASS_NAMES[idx_sorted[0]]
    confidence   = float(probs[idx_sorted[0]])

    top_k_results = [
        {
            'class':      CLASS_NAMES[i],
            'confidence': float(probs[i]),
            'info':       CLASS_INFO[CLASS_NAMES[i]],
        }
        for i in idx_sorted[:top_k]
    ]

    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        'predicted_class':   top_class,
        'confidence':        confidence,
        'top_k':             top_k_results,
        'all_probabilities': all_probs,
        'info':              CLASS_INFO[top_class],
    }


# ─── BATCH PREDICTION ────────────────────────────────────────────────────────
def predict_batch(image_paths: list, model_path: str = None) -> list:
    """Predict for a list of image file paths."""
    model = load_model(model_path)
    results = []
    batch  = np.vstack([preprocess_image(p) for p in image_paths])

    if model is not None:
        probs_all = model.predict(batch, verbose=0)
    else:
        probs_all = np.array([
            np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
            for _ in image_paths
        ])

    for i, path in enumerate(image_paths):
        probs    = probs_all[i]
        top_idx  = int(np.argmax(probs))
        top_cls  = CLASS_NAMES[top_idx]
        results.append({
            'image_path':    path,
            'predicted_class': top_cls,
            'confidence':    float(probs[top_idx]),
            'info':          CLASS_INFO[top_cls],
        })
    return results


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        result = predict(path)
        print(f"\nPrediction : {result['predicted_class']}")
        print(f"Confidence : {result['confidence']*100:.1f}%")
        print(f"Risk Level : {result['info']['risk']}")
        print(f"Action     : {result['info']['action']}")
    else:
        print("Usage: python predict.py <image_path>")
