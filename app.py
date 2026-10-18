"""
Skin Disease Classification - Flask Web Application
"""

import os
import io
import base64
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict, CLASS_NAMES, CLASS_INFO

# ─── APP CONFIG ──────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'),
)

UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img: Image.Image, fmt: str = 'JPEG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES, class_info=CLASS_INFO)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST /api/predict
    Accepts multipart/form-data with key 'image'.
    Returns JSON with prediction results.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, BMP, TIFF, or WEBP.'}), 400

    try:
        # Read image bytes
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Save upload (with UUID to avoid conflicts)
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(save_path)

        # Run prediction
        result = predict(img_bytes, top_k=3)

        # Prepare thumbnail for response
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        thumb_b64 = image_to_base64(thumb)

        response = {
            'success':         True,
            'predicted_class': result['predicted_class'],
            'confidence':      round(result['confidence'] * 100, 2),
            'risk':            result['info']['risk'],
            'description':     result['info']['description'],
            'action':          result['info']['action'],
            'short':           result['info']['short'],
            'color':           result['info']['color'],
            'top_k': [
                {
                    'class':      r['class'],
                    'confidence': round(r['confidence'] * 100, 2),
                    'short':      r['info']['short'],
                    'risk':       r['info']['risk'],
                    'color':      r['info']['color'],
                }
                for r in result['top_k']
            ],
            'all_probabilities': {
                k: round(v * 100, 2)
                for k, v in result['all_probabilities'].items()
            },
            'thumbnail': f'data:image/jpeg;base64,{thumb_b64}',
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/classes')
def api_classes():
    """Return all supported disease classes with metadata."""
    return jsonify({
        'classes': [
            {
                'name':        name,
                'short':       info['short'],
                'description': info['description'],
                'risk':        info['risk'],
                'action':      info['action'],
                'color':       info['color'],
            }
            for name, info in CLASS_INFO.items()
        ]
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─── ERROR HANDLERS ──────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10 MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"\n🩺  Skin Disease Classifier running on http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
