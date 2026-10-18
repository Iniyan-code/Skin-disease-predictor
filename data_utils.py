"""
Skin Disease Classification - Data Utilities
Handles HAM10000 dataset download, preprocessing, and EDA.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

LABEL_MAP = {
    'nv':   'Melanocytic Nevi',
    'mel':  'Melanoma',
    'bkl':  'Benign Keratosis',
    'bcc':  'Basal Cell Carcinoma',
    'akiec':'Actinic Keratosis',
    'vasc': 'Vascular Lesions',
    'df':   'Dermatofibroma',
}

CLASS_NAMES = list(LABEL_MAP.values())


# ─── DATASET EDA ─────────────────────────────────────────────────────────────
def run_eda(csv_path: str, image_dir: str, output_dir: str = 'eda_output'):
    """Generate exploratory data analysis plots for HAM10000."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['label_name'] = df['dx'].map(LABEL_MAP)

    # ── Class distribution ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    counts = df['label_name'].value_counts()

    axes[0].bar(range(len(counts)), counts.values,
                color=plt.cm.Set3(np.linspace(0, 1, len(counts))))
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=45, ha='right')
    axes[0].set_title('Class Distribution (Count)')
    axes[0].set_ylabel('Number of Images')

    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=plt.cm.Set3(np.linspace(0, 1, len(counts))))
    axes[1].set_title('Class Distribution (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/class_distribution.png")

    # ── Age distribution by class ──────────────────────────────────────────
    if 'age' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='label_name', y='age', palette='Set3')
        plt.xticks(rotation=45, ha='right')
        plt.title('Age Distribution by Skin Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/age_distribution.png")

    # ── Sex distribution ───────────────────────────────────────────────────
    if 'sex' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['sex'].value_counts().plot.bar(ax=axes[0], color=['#64b5f6', '#f48fb1'])
        axes[0].set_title('Overall Sex Distribution')
        axes[0].set_xlabel('')

        sex_class = pd.crosstab(df['label_name'], df['sex'])
        sex_class.plot.bar(ax=axes[1], stacked=True, color=['#64b5f6', '#f48fb1'])
        axes[1].set_title('Sex Distribution by Class')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sex_distribution.png'), dpi=150)
        plt.close()

    # ── Localization ───────────────────────────────────────────────────────
    if 'localization' in df.columns:
        loc_counts = df['localization'].value_counts().head(15)
        plt.figure(figsize=(10, 6))
        loc_counts.plot.barh(color=plt.cm.viridis(np.linspace(0, 1, len(loc_counts))))
        plt.title('Top 15 Lesion Localizations')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'localization.png'), dpi=150)
        plt.close()

    # ── Sample images ──────────────────────────────────────────────────────
    _plot_sample_images(df, image_dir, output_dir)

    print(f"\nEDA complete. All plots saved to: {output_dir}/")
    print(f"\nDataset Summary:\n{df['label_name'].value_counts().to_string()}")


def _plot_sample_images(df: pd.DataFrame, image_dir: str, output_dir: str):
    """Plot one sample image per class."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (cls_name) in enumerate(CLASS_NAMES):
        subset = df[df['label_name'] == cls_name]
        if len(subset) == 0:
            continue
        row = subset.sample(1).iloc[0]
        img_path = None
        for sub in ['HAM10000_images_part_1', 'HAM10000_images_part_2', '']:
            p = os.path.join(image_dir, sub, f"{row['image_id']}.jpg")
            if os.path.exists(p):
                img_path = p
                break

        if img_path and idx < len(axes):
            img = Image.open(img_path).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].set_title(cls_name, fontsize=9, wrap=True)
            axes[idx].axis('off')

    # Hide unused subplots
    for i in range(len(CLASS_NAMES), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Images per Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/sample_images.png")


# ─── DATASET PREPARATION ─────────────────────────────────────────────────────
def balance_dataset(df: pd.DataFrame, strategy: str = 'oversample',
                    target_count: int = 1500) -> pd.DataFrame:
    """
    Balance the dataset.
    strategy: 'oversample' | 'undersample'
    """
    frames = []
    for cls in CLASS_NAMES:
        subset = df[df['label_name'] == cls]
        n = len(subset)

        if strategy == 'oversample' and n < target_count:
            extra = subset.sample(target_count - n, replace=True, random_state=42)
            frames.append(pd.concat([subset, extra]))
        elif strategy == 'undersample' and n > target_count:
            frames.append(subset.sample(target_count, random_state=42))
        else:
            frames.append(subset)

    balanced = pd.concat(frames).reset_index(drop=True)
    print(f"\nBalanced dataset ({strategy}):")
    print(balanced['label_name'].value_counts().to_string())
    return balanced


def compute_normalization_stats(image_paths: list, sample_size: int = 1000) -> dict:
    """Compute per-channel mean and std for normalization."""
    if len(image_paths) > sample_size:
        image_paths = np.random.choice(image_paths, sample_size, replace=False)

    means, stds = [], []
    for p in image_paths:
        try:
            arr = np.array(Image.open(p).convert('RGB').resize((224, 224))) / 255.0
            means.append(arr.mean(axis=(0, 1)))
            stds.append(arr.std(axis=(0, 1)))
        except Exception:
            continue

    channel_mean = np.mean(means, axis=0)
    channel_std  = np.mean(stds, axis=0)
    print(f"Channel mean: {channel_mean}")
    print(f"Channel std:  {channel_std}")
    return {'mean': channel_mean.tolist(), 'std': channel_std.tolist()}


# ─── DOWNLOAD INSTRUCTIONS ───────────────────────────────────────────────────
def print_download_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          HAM10000 Dataset Download Instructions              ║
╠══════════════════════════════════════════════════════════════╣
║  1. Create a Kaggle account at https://www.kaggle.com        ║
║  2. Go to: kaggle.com/datasets/kmader/skin-cancer-mnist-     ║
║     ham10000                                                 ║
║  3. Click "Download" and extract to your data/ folder        ║
║                                                              ║
║  OR via Kaggle CLI:                                          ║
║    pip install kaggle                                        ║
║    kaggle datasets download kmader/skin-cancer-mnist-        ║
║      ham10000 -p data/                                       ║
║    unzip data/skin-cancer-mnist-ham10000.zip -d data/        ║
║                                                              ║
║  Expected structure:                                         ║
║    data/                                                     ║
║      HAM10000_metadata.csv                                   ║
║      HAM10000_images_part_1/  (*.jpg)                        ║
║      HAM10000_images_part_2/  (*.jpg)                        ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eda',        action='store_true', help='Run EDA')
    parser.add_argument('--csv',        default='data/HAM10000_metadata.csv')
    parser.add_argument('--image_dir',  default='data/')
    parser.add_argument('--output_dir', default='eda_output')
    parser.add_argument('--instructions', action='store_true')
    args = parser.parse_args()

    if args.instructions:
        print_download_instructions()
    elif args.eda:
        run_eda(args.csv, args.image_dir, args.output_dir)
    else:
        print_download_instructions()
