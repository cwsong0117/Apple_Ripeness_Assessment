"""
train_final_model.py
---------------------
Trains the FINAL deployable SVM model on 100% of the dataset, using the
hyperparameters already found via GridSearchCV in SVM.py.

Why this is separate from SVM.py:
    SVM.py's job was to EVALUATE the model (train/test split, cross
    validation, hyperparameter search) - it deliberately holds back data
    so it can report honest performance numbers.

    This script's job is different: now that model selection and evaluation
    are done, we retrain on ALL available images (no held-out split) using
    the hyperparameters already chosen, so the final deployed model learns
    from every image you have. This is standard practice once you're done
    comparing/justifying model choice and are ready to ship.

    This script does NOT report a test accuracy, because it has no held-out
    test set left - that's expected and correct. Your reported accuracy
    figures should come from SVM.py's evaluation runs, not from this script.

What the model is trained on:
    By default each training photo is first run through the app's own detector
    (UI/image_pipeline.detect_apples) and the model learns from the crop the
    app would classify. At prediction time the app only ever sees detector
    crops, not whole photos, so training on whole photos teaches features such
    as a stalk or a dried calyx tail that the detector now trims off. If the
    detector finds nothing in a photo, the whole photo is used instead.
    Pass --whole-images to reproduce the older whole-photo training.

Output:
    svm_final_model.pkl - saved via joblib, contains the fitted pipeline
    (scaler + SVM) and the label encoder, ready to be loaded by predict.py.

Requirements:
    pip install opencv-python numpy scikit-learn scikit-image joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Setup: import the shared preprocessing module (searches upward)
# ---------------------------------------------------------------------------
import importlib


def find_preprocessing_module():
    """
    Locate preprocessing.py by searching upward from this script's location
    (and recursively downward at each level), then import it correctly
    whether it is a standalone file or part of a real package (e.g.
    "module/Module1/" with its own __init__.py, config.py, model.py, etc,
    using relative imports internally like "from .config import ...").

    A plain sys.path.append() + "from preprocessing import ..." only works
    for standalone files - if preprocessing.py belongs to a package, that
    approach breaks its internal relative imports with:
        ImportError: attempted relative import with no known parent package

    This function handles both cases automatically.
    """
    start = Path(__file__).resolve().parent
    preprocessing_path = None
    for parent in [start] + list(start.parents):
        matches = list(parent.rglob("preprocessing.py"))
        if matches:
            preprocessing_path = matches[0]
            break

    if preprocessing_path is None:
        raise FileNotFoundError(
            f"Could not find a 'preprocessing.py' file anywhere above {start}. "
            f"Check your folder structure."
        )

    package_dir = preprocessing_path.parent
    package_name = package_dir.name
    parent_of_package = str(package_dir.parent)

    # Try package-style import FIRST (e.g. "Module1.preprocessing"). This
    # works whether or not Module1 has an __init__.py - Python 3 supports
    # "namespace packages" without one - and is REQUIRED if preprocessing.py
    # uses relative imports internally (e.g. "from .config import ...").
    if parent_of_package not in sys.path:
        sys.path.insert(0, parent_of_package)
    try:
        return importlib.import_module(f"{package_name}.preprocessing")
    except ImportError:
        pass

    # Fall back to flat-file import (old style, no internal relative imports).
    dir_str = str(package_dir)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)
    return importlib.import_module("preprocessing")


_preprocessing = find_preprocessing_module()
segment_background = _preprocessing.segment_background
apply_clahe = _preprocessing.apply_clahe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parent

# Relative to the repository so the script runs on any machine.
DATASET_DIR = REPO_ROOT / "Apple Ripeness Levels Image Dataset"

# OPTIONAL "Not Apple" class. Put photos of things the detector must NOT report
# as apples (leaves, branches, hands, empty backgrounds, other fruit) in this
# folder. It is deliberately kept OUTSIDE DATASET_DIR so the six-class
# evaluation scripts (SVM.py, RandomForest.py, CNN.py, compare_model.py) and
# the accuracy figures reported from them are unchanged.
# Must match NOT_APPLE_LABEL in module/classification.py.
NEGATIVES_DIR = OUTPUT_DIR / "not_apple_images"
NOT_APPLE_LABEL = "Not Apple"
MIN_NEGATIVES_WARNING = 60
NOT_APPLE_REJECT_PROBABILITY = 0.5  # same rule the app uses
FINAL_MODEL_PATH = OUTPUT_DIR / "svm_final_model.pkl"  # saved in module/ so predict.py finds it easily

IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_STATE = 42

# Hyperparameters already found via GridSearchCV in SVM.py - fill in your
# actual best params here if different.
SVM_BEST_PARAMS = {"C": 10, "gamma": 0.01, "kernel": "rbf"}


# ---------------------------------------------------------------------------
# Feature extraction (identical to SVM.py - must never drift from this)
# ---------------------------------------------------------------------------
def extract_color_features(image: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    stats = np.array([
        hsv[:, :, 0].mean(), hsv[:, :, 0].std(),
        hsv[:, :, 1].mean(), hsv[:, :, 1].std(),
        hsv[:, :, 2].mean(), hsv[:, :, 2].std(),
    ])

    return np.concatenate([hist, stats])


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return np.array([
        graycoprops(glcm, "contrast").mean(),
        graycoprops(glcm, "dissimilarity").mean(),
        graycoprops(glcm, "homogeneity").mean(),
        graycoprops(glcm, "energy").mean(),
        graycoprops(glcm, "correlation").mean(),
        graycoprops(glcm, "ASM").mean(),
    ])


def extract_features(image: np.ndarray) -> np.ndarray:
    return np.concatenate([extract_color_features(image), extract_texture_features(image)])


def preprocess_image(image: np.ndarray) -> np.ndarray:
    return apply_clahe(segment_background(image))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.resize(image, IMAGE_SIZE)


def detector_crop(path: Path) -> np.ndarray | None:
    """Return the crop the app would classify for this photo.

    Uses the same detect-and-crop functions as the app, so training crops
    cannot drift from served crops. Returns None when the detector finds no
    apple (or the file cannot be read). With several detections the largest is
    used; each dataset photo shows one ripeness level.
    """
    # Imported here so --whole-images does not need the app's modules.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from UI.image_pipeline import crop_apple, decode_image, detect_apples

    try:
        preprocessing, _, apples, _, _ = detect_apples(decode_image(path.read_bytes()))
    except Exception:
        return None
    if not apples:
        return None

    largest = max(apples, key=lambda apple: apple["bbox"][2] * apple["bbox"][3])
    return crop_apple(preprocessing, tuple(largest["bbox"]))


def crop_features(crop: np.ndarray) -> np.ndarray:
    """Features of a detector crop, computed exactly as module/classification.py does."""
    resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return extract_features(preprocess_image(resized))


def load_dataset(dataset_dir: Path, use_crops: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    X, y = [], []
    for class_dir in class_dirs:
        _load_class_images(class_dir, class_dir.name, X, y, use_crops)

    return np.array(X), np.array(y)


def _load_class_images(class_dir: Path, label: str, X: list, y: list,
                       use_crops: bool = True) -> int:
    """Extract features for every image in ``class_dir`` and append them with ``label``."""
    image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]

    loaded = cropped = 0
    for img_path in image_paths:
        try:
            crop = detector_crop(img_path) if use_crops else None
            if crop is not None:
                features = crop_features(crop)
                cropped += 1
            else:
                image = load_image(img_path)
                if image is None:
                    continue
                features = extract_features(preprocess_image(image))
        except Exception:
            continue
        X.append(features)
        y.append(label)
        loaded += 1

    detail = f" ({cropped} detector crops, {loaded - cropped} whole photos)" if use_crops else ""
    print(f"  Class '{label}': {loaded}/{len(image_paths)} images{detail}")
    return loaded


def add_negatives(X: np.ndarray, y: np.ndarray,
                  use_crops: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Append the optional "Not Apple" images, if the folder has any."""
    if not NEGATIVES_DIR.exists():
        print(f"No '{NOT_APPLE_LABEL}' folder ({NEGATIVES_DIR.name}) - training six ripeness classes only.")
        return X, y

    neg_X, neg_y = [], []
    count = _load_class_images(NEGATIVES_DIR, NOT_APPLE_LABEL, neg_X, neg_y, use_crops)
    if count == 0:
        print(f"'{NEGATIVES_DIR.name}' has no usable images - training six ripeness classes only.")
        return X, y

    if count < MIN_NEGATIVES_WARNING:
        print(f"  WARNING: only {count} '{NOT_APPLE_LABEL}' images. "
              f"Aim for roughly 100 (one apple class has ~100) or the class will be unreliable.")

    return np.vstack([X, np.array(neg_X)]), np.concatenate([y, np.array(neg_y)])


def report_not_apple_quality(pipeline: Pipeline, X: np.ndarray, y_encoded: np.ndarray,
                             encoder: LabelEncoder) -> None:
    """Cross-validated check of the apple / not-apple decision the app will make.

    The final model is fit on everything, so it cannot be scored on its own
    training data. This 5-fold check uses the same rule as the app: reject when
    P("Not Apple") >= NOT_APPLE_REJECT_PROBABILITY.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    not_apple_index = list(encoder.classes_).index(NOT_APPLE_LABEL)
    is_negative = y_encoded == not_apple_index
    folds = min(5, int(is_negative.sum()), int((~is_negative).sum()))
    if folds < 2:
        print("Too few samples for a cross-validated Not Apple check.")
        return

    probabilities = cross_val_predict(
        pipeline, X, y_encoded,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )
    rejected = probabilities[:, not_apple_index] >= NOT_APPLE_REJECT_PROBABILITY

    caught = rejected[is_negative].mean()
    wrongly_rejected = rejected[~is_negative].mean()
    print(f"\n{folds}-fold cross-validated Not Apple check (app's rejection rule):")
    print(f"  Non-apples correctly rejected : {caught:.1%}  ({int(rejected[is_negative].sum())}/{int(is_negative.sum())})")
    print(f"  Real apples wrongly rejected  : {wrongly_rejected:.1%}  ({int(rejected[~is_negative].sum())}/{int((~is_negative).sum())})")
    print("  Rule of thumb: keep wrongly-rejected apples under ~3%; if not, add more varied")
    print("  Not Apple photos (especially ones that look like your apple photos).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the final deployable SVM.")
    parser.add_argument(
        "--whole-images", action="store_true",
        help="train on whole photos instead of the detector's crops (the older behaviour)",
    )
    parser.add_argument(
        "--output", type=Path, default=FINAL_MODEL_PATH,
        help=f"where to save the model (default: {FINAL_MODEL_PATH.name} next to this script)",
    )
    args = parser.parse_args()
    use_crops = not args.whole_images

    source = "detector crops" if use_crops else "whole photos"
    print(f"Loading FULL dataset as {source} (no train/test split - this is the final model)...")
    X, y = load_dataset(DATASET_DIR, use_crops)
    X, y = add_negatives(X, y, use_crops)
    has_negatives = NOT_APPLE_LABEL in set(y)
    print(f"\nTotal samples used for final training: {len(X)}\n")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=SVM_BEST_PARAMS["C"],
            gamma=SVM_BEST_PARAMS["gamma"],
            kernel=SVM_BEST_PARAMS["kernel"],
            probability=True,   # needed so predict.py can return confidence scores
            random_state=RANDOM_STATE,
        )),
    ])

    if has_negatives:
        report_not_apple_quality(pipeline, X, y_encoded, encoder)

    print("\nTraining final SVM on 100% of the dataset...")
    pipeline.fit(X, y_encoded)

    joblib.dump({"model": pipeline, "label_encoder": encoder}, args.output)
    print(f"\nFinal deployable model saved to: {args.output}")
    print("This model has seen every image in the dataset and is ready for predict.py.")


if __name__ == "__main__":
    main()