import kagglehub
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers
import tensorflow as tf

# ── 1. Download dataset ───────────────────────────────────────────────────
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Dataset path:", path)

CSV_PATH  = os.path.join(path, "HAM10000_metadata.csv")
IMAGE_DIR = path

# ── 2. Label mapping ──────────────────────────────────────────────────────
LABEL_MAP = {
    'nv': 'Melanocytic Nevi', 'mel': 'Melanoma',
    'bkl': 'Benign Keratosis', 'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis', 'vasc': 'Vascular Lesions', 'df': 'Dermatofibroma'
}
CLASS_NAMES = list(LABEL_MAP.values())

# ── 3. Load metadata ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['label_name'] = df['dx'].map(LABEL_MAP)
df['label_idx']  = pd.Categorical(df['label_name'], categories=CLASS_NAMES).codes

def find_image(image_id):
    for sub in ['HAM10000_images_part_1', 'HAM10000_images_part_2', '']:
        p = os.path.join(IMAGE_DIR, sub, f'{image_id}.jpg')
        if os.path.exists(p):
            return p
    return None

df['filepath'] = df['image_id'].apply(find_image)
df = df.dropna(subset=['filepath'])
print(f"Loaded {len(df)} images")
print(df['label_name'].value_counts())

# ── 4. Split ──────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'], random_state=42)
val_df, test_df  = train_test_split(val_df, test_size=0.5, stratify=val_df['label_idx'], random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 5. Generators ─────────────────────────────────────────────────────────
IMG_SIZE, BATCH = 224, 32

train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=30,
    horizontal_flip=True, vertical_flip=True, zoom_range=0.2
).flow_from_dataframe(train_df, x_col='filepath', y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=True)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val_df, x_col='filepath', y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, x_col='filepath', y_col='label_name',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False)

# ── 6. Class weights ──────────────────────────────────────────────────────
weights = compute_class_weight('balanced',
    classes=np.unique(train_df['label_idx']),
    y=train_df['label_idx'].values)
class_weights = dict(enumerate(weights))

# ── 7. Model ──────────────────────────────────────────────────────────────
base = EfficientNetB3(weights='imagenet', include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(512, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ── 8. Train ──────────────────────────────────────────────────────────────
model.fit(
    train_gen, validation_data=val_gen,
    epochs=20, class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.h5',
            save_best_only=True, monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
            patience=3, verbose=1),
    ]
)

# ── 9. Evaluate & save ────────────────────────────────────────────────────
results = model.evaluate(test_gen)
print(f"\nTest Accuracy: {results[1]*100:.2f}%")
model.save('skin_disease_model.h5')
print("Saved: skin_disease_model.h5")