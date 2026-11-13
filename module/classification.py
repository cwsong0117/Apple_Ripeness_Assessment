from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "Model Training" / "svm_final_model.pkl"
IMAGE_SIZE = (256, 256)

# Optional seventh class. When the final model is trained with photos of leaves,
# hands, branches and backgrounds (see Model Training/not_apple_images), the
# SVM can answer "this is not an apple" instead of forcing one of the six
# ripeness labels onto it. Must match NOT_APPLE_LABEL in train_final_model.py.
# A model trained without that class simply never produces it, so behaviour is
# unchanged until the model is retrained.
NOT_APPLE_LABEL = "Not Apple"
# Reject when the model puts at least this much probability on "Not Apple".
# Comparing against the whole apple side (1 - p) rather than the single best
# ripeness class matters: a leaf's apple probability is split across six
# ripeness classes, so no single one of them would ever beat "Not Apple".
NOT_APPLE_REJECT_PROBABILITY = 0.5


def _validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("predict_ripeness expects a BGR image with 3 channels.")


def _prepare_image(image: np.ndarray, fast: bool = False) -> np.ndarray:
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if fast:
        processed = preprocess_image_fast(resized)
    else:
        processed = preprocess_image(resized)
    return extract_features(processed).reshape(1, -1)


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
    from module.Module1.preprocessing import apply_clahe, segment_background

    return apply_clahe(segment_background(image))


def preprocess_image_fast(image: np.ndarray) -> np.ndarray:
    """Approximate the training preprocessing without iterative GrabCut.

    Live-camera ROIs have already been localized by the detector, so the HSV
    candidate mask is sufficient to remove most of the background.  Avoiding
    GrabCut here makes live inference several times faster while preserving
    the feature layout expected by the trained SVM.
    """
    from module.Module1.preprocessing import apply_clahe, create_apple_candidate_mask

    candidate_mask = create_apple_candidate_mask(image)
    segmented = cv2.bitwise_and(image, image, mask=candidate_mask)
    return apply_clahe(segmented)


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model file not found: {MODEL_PATH}")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"] if isinstance(payload, dict) else payload
    encoder = payload["label_encoder"] if isinstance(payload, dict) else None
    if encoder is None:
        raise ValueError(f"Label encoder missing from SVM model file: {MODEL_PATH}")
    return model, encoder


def predict_ripeness(image: np.ndarray, *, fast: bool = False) -> dict:
    """Predict ripeness, optionally using the low-latency live-camera path."""
    _validate_image(image)
    model, encoder = _load_artifacts()

    model_input = _prepare_image(image, fast=fast)
    probabilities = model.predict_proba(model_input)[0]
    classes = list(encoder.classes_)

    class_probabilities = {
        label: float(probabilities[index]) for index, label in enumerate(classes)
    }

    # Split off the optional "Not Apple" class so the label, confidence and
    # probabilities seen by the UI and report stay purely about ripeness.
    has_not_apple_class = NOT_APPLE_LABEL in class_probabilities
    not_apple_probability = class_probabilities.pop(NOT_APPLE_LABEL, 0.0)
    is_apple = not_apple_probability < NOT_APPLE_REJECT_PROBABILITY

    if has_not_apple_class:
        # Renormalise over the ripeness classes so they again sum to 1.
        apple_total = sum(class_probabilities.values())
        if apple_total > 0:
            class_probabilities = {
                label: probability / apple_total
                for label, probability in class_probabilities.items()
            }

    predicted_label = max(class_probabilities, key=class_probabilities.get)
    confidence = class_probabilities[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": class_probabilities,
        "is_apple": is_apple,
        "not_apple_probability": not_apple_probability,
    }
