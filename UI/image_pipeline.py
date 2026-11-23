"""Bridge Streamlit image input to the three image-processing modules.

``interface.py`` supplies camera/upload bytes to :func:`analyse_image`. This
module validates and decodes those bytes, runs preprocessing, asks the object
detector for apple bounding boxes, and sends every apple crop to the trained
ripeness classifier. It then packages everything into dataclasses that the UI
can display and ``analysis_repository.py`` can save.

This file does not train the model. The trained SVM is loaded inside
``module.classification.predict_ripeness`` when a prediction is requested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.Module1.model import PreprocessingResult
from module.Module1.preprocessing import preprocess_image_with_steps
from module.object_detection import annotate_apples, detect_objects


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass
class AppleResult:
    """Classification output and display data for one detected apple."""

    apple_id: int
    crop: np.ndarray
    label: str
    confidence: float
    probabilities: dict[str, float]
    bbox: tuple[int, int, int, int]
    classification_error: str | None = None


@dataclass
class ImageAnalysis:
    """Complete result for one uploaded or camera-captured image."""

    source: np.ndarray
    processed: np.ndarray
    annotated: np.ndarray
    preprocessing: PreprocessingResult
    apples: list[AppleResult]
    detection_message: str


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a three-channel OpenCV BGR image."""
    if not image_bytes:
        raise ValueError("The selected image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError("The selected image is larger than the 10 MB limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            corrected = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb_image = np.asarray(corrected)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file could not be read as an image.") from exc

    if rgb_image.shape[0] < 100 or rgb_image.shape[1] < 100:
        raise ValueError("The image must be at least 100 × 100 pixels.")

    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def _white_segmented_background(
    image: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """Copy the foreground onto white so detection sees less background noise."""
    white_background = np.full_like(image, 255)
    return cv2.copyTo(image, foreground_mask, white_background)


def detect_apples(
    source: np.ndarray,
) -> tuple[PreprocessingResult, np.ndarray, list[dict], np.ndarray, str]:
    """Run Module 1 preprocessing and Module 2 detection on a decoded image.

    Returns ``(preprocessing, processed, detections, annotated, message)``.
    Kept separate from :func:`analyse_image` so that
    ``Model Training/train_final_model.py`` can build its training crops with
    exactly the code the app uses, instead of a copy that could drift.
    """
    # Module 1: create every preprocessing/segmentation stage used by the UI's
    # stage viewer.
    preprocessing = preprocess_image_with_steps(source)
    processed = preprocessing.final

    if preprocessing.segmentation_success:
        processed = _white_segmented_background(
            processed,
            preprocessing.refined_mask,
        )

    # Module 2: locate apples and return a bounding box for each one, plus an
    # annotated image that can be displayed in the result panel.
    detected_apples, annotated, _mask, message = detect_objects(processed)
    return preprocessing, processed, detected_apples, annotated, message


def crop_apple(
    preprocessing: PreprocessingResult,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Cut one detection out of the clean, resized image for classification.

    Classification performs its own preprocessing, so the crop comes from the
    resized image rather than the processed one: contrast enhancement and
    segmentation are then not accidentally applied twice.
    """
    x, y, width, height = bbox
    return np.ascontiguousarray(preprocessing.resized[y:y + height, x:x + width])


def analyse_image(image_bytes: bytes) -> ImageAnalysis:
    """Run preprocessing, detection, and classification on one image."""
    # Modules 1 and 2: decode the browser bytes, then preprocess and detect.
    source = decode_image(image_bytes)
    preprocessing, processed, detected_apples, annotated, message = detect_apples(source)
    apple_results: list[AppleResult] = []
    kept_detections: list[dict] = []
    rejected_count = 0

    # Module 3 is imported here rather than at application startup. If the
    # saved model or one of its dependencies is missing, detection can still
    # succeed and the UI can show a clear classification error.
    predict_ripeness = None
    classifier_import_error = None
    try:
        from module.classification import predict_ripeness
    except (ImportError, ValueError) as exc:
        classifier_import_error = (
            "The classification dependencies could not be loaded: "
            f"{exc}"
        )

    for detected_apple in detected_apples:
        x, y, width, height = detected_apple["bbox"]
        apple_id = int(detected_apple["id"])

        # Classification performs its own preprocessing (see crop_apple).
        crop = crop_apple(preprocessing, (x, y, width, height))

        label = "Unavailable"
        confidence = 0.0
        probabilities: dict[str, float] = {}
        classification_error = classifier_import_error

        if predict_ripeness is not None:
            try:
                # This is the exact call that reaches the teammate-trained SVM
                # in module/classification.py. It returns the winning class,
                # confidence, and probabilities for all six ripeness classes.
                prediction = predict_ripeness(crop)

                # A model trained with a "Not Apple" class can recognise a
                # leaf, hand or branch the colour/shape tests let through.
                # Discard it rather than reporting an invented ripeness.
                if not prediction.get("is_apple", True):
                    rejected_count += 1
                    continue

                label = str(prediction["label"])
                confidence = float(prediction["confidence"])
                probabilities = {
                    str(class_label): float(probability)
                    for class_label, probability in prediction.get(
                        "probabilities", {}
                    ).items()
                }
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
                classification_error = str(exc)

        apple_results.append(
            AppleResult(
                apple_id=apple_id,
                crop=crop,
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                bbox=(x, y, width, height),
                classification_error=classification_error,
            )
        )
        kept_detections.append(detected_apple)

    # Some detections were discarded as "not an apple": renumber the survivors
    # and redraw the overlay so the picture and the results stay consistent.
    if rejected_count:
        for new_id, (detection, result) in enumerate(
            zip(kept_detections, apple_results), start=1
        ):
            detection["id"] = new_id
            result.apple_id = new_id
        annotated = annotate_apples(processed, kept_detections)

        ignored = f"{rejected_count} object(s) ignored: not recognised as an apple."
        if apple_results:
            message = f"Apple detected! Found {len(apple_results)} apple(s). {ignored}"
        else:
            message = f"No apple found in the image. {ignored}"

    # Return one structured object instead of letting the UI depend directly
    # on the internal outputs of every processing module.
    return ImageAnalysis(
        source=source,
        processed=processed,
        annotated=annotated,
        preprocessing=preprocessing,
        apples=apple_results,
        detection_message=message,
    )
