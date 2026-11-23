import numpy as np
import cv2 as cv

# ==========================================
# 0. SHARED APPLE-SHAPE TEST
# ==========================================
# Colour alone cannot separate a green apple from a leaf (both fall in the same
# HSV range), so a candidate must also be compact. A leaf is a long, pointed
# oval, which the elongation limit below rejects. Both the still-image detector
# and the live-camera detector use this one test so they cannot drift apart.
# Known limit: a leaf that is nearly as wide as it is long (elongation < 1.7)
# still passes; only a "not apple" training class can fix that.
#
# Thresholds were calibrated on the dataset's apples (88 sampled images):
# elongation median 1.08 / 95th percentile 1.45, circularity minimum 0.33,
# solidity minimum 0.71, extent minimum 0.43. Circularity and solidity cannot
# separate a smooth oval leaf from an apple (a 2:1 ellipse still scores ~0.8
# circularity), so the leaf-specific test is MAX_ELONGATION.
MIN_ASPECT = 0.50
MAX_ASPECT = 1.60
MAX_ELONGATION = 1.70
MIN_SOLIDITY = 0.65
MIN_CIRCULARITY = 0.30
MIN_EXTENT = 0.40
# A round apple fills about pi/4 = 0.785 of its bounding box (dataset maximum
# 0.80). A blob that fills nearly all of it is rectangular - a table, tray or
# unsegmented background - not an apple.
MAX_EXTENT = 0.90
# Close-up photos legitimately fill most of the frame (a round apple can reach
# ~78% of a square image), so only reject a blob that is practically the whole
# image.
MIN_AREA_FRACTION = 0.005
MAX_AREA_FRACTION = 0.95


def compute_shape_metrics(contour: np.ndarray) -> dict:
    """Measure the shape descriptors used to decide whether a blob is an apple."""
    area = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    perimeter = cv.arcLength(contour, True)
    hull_area = cv.contourArea(cv.convexHull(contour))

    # Elongation = long side / short side of the minimum-area rotated box.
    # Unlike the axis-aligned aspect ratio it does not change when the object
    # is tilted, so a leaf lying diagonally is still recognised as long and thin.
    (_, _), (rect_w, rect_h), _ = cv.minAreaRect(contour)
    short_side, long_side = sorted((rect_w, rect_h))
    elongation = long_side / short_side if short_side > 0 else float("inf")

    return {
        "area": area,
        "elongation": elongation,
        "aspect": w / float(h) if h > 0 else 0.0,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "circularity": (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0,
        "extent": area / float(w * h) if w > 0 and h > 0 else 0.0,
    }


def is_apple_shape(
    metrics: dict,
    *,
    min_solidity: float = MIN_SOLIDITY,
    min_circularity: float = MIN_CIRCULARITY,
    min_extent: float = MIN_EXTENT,
) -> bool:
    """Return True when a blob is compact and round enough to be an apple."""
    return (
        MIN_ASPECT <= metrics["aspect"] <= MAX_ASPECT
        and metrics["elongation"] <= MAX_ELONGATION
        and metrics["solidity"] >= min_solidity
        and metrics["circularity"] >= min_circularity
        and min_extent <= metrics["extent"] <= MAX_EXTENT
    )


# ==========================================
# 0b. CONTOUR-DETECTION HELPERS
# ==========================================
# Pixels whose weakest channel is at least this are treated as white background.
WHITE_LEVEL = 250
# An object is trimmed with a disc this fraction of its own inscribed radius.
# A round apple survives it almost unchanged; anything narrower than about
# 1.1x the apple's radius (stalk, stick, dried calyx tail, thin leaf) does not.
TRIM_RADIUS_RATIO = 0.55
# Ignore blobs smaller than this fraction of the image before trimming.
MIN_COMPONENT_FRACTION = 0.002
# Watershed seeds are the pixels within this fraction of each blob's own peak
# distance-to-edge (measured per blob, not across the whole image).
SEED_FRACTION = 0.8


def _has_white_background(image: np.ndarray) -> bool:
    """True when most of the image border is white, i.e. it is already segmented."""
    height, width = image.shape[:2]
    band = max(2, int(0.03 * min(height, width)))
    ring = np.ones((height, width), dtype=bool)
    ring[band:-band, band:-band] = False
    return bool((image.min(axis=2)[ring] >= WHITE_LEVEL).mean() >= 0.5)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (specular highlights, pale patches) in a binary mask."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, 255, thickness=cv.FILLED)
    return filled


def _candidate_mask(image: np.ndarray) -> np.ndarray:
    """Foreground mask of everything that could be an apple.

    After segmentation the background is exactly white, so "not white" is the
    most faithful foreground: unlike a saturation test it keeps pale-green and
    bright highlight regions of the apple, which otherwise punch holes in or
    bite chunks out of the contour. For an unsegmented image it falls back to a
    colour (saturation/brightness) test.
    """
    smoothed = cv.medianBlur(image, 5)  # (Consider increasing to 15 if water droplets cause issues again)
    if _has_white_background(smoothed):
        mask = np.where(smoothed.min(axis=2) < WHITE_LEVEL, 255, 0).astype(np.uint8)
    else:
        hsv = cv.cvtColor(smoothed, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return _fill_holes(mask)


def trim_thin_attachments(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Remove stalks, sticks, dried tails and thin leaves joined to a blob.

    Each connected blob is morphologically opened with a disc sized relative to
    that blob's own largest inscribed circle, so the amount of trimming scales
    with the apple instead of being a fixed pixel count.
    """
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    trimmed = np.zeros_like(mask)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(mask.shape[1], x + w + 2), min(mask.shape[0], y + h + 2)
        blob = (labels[y0:y1, x0:x1] == index).astype(np.uint8) * 255

        # Pad with background only to measure the size: a blob that fills the
        # whole region (e.g. a plain wall) has no background pixel to measure to.
        padded = cv.copyMakeBorder(blob, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
        radius = int(round(TRIM_RADIUS_RATIO * cv.distanceTransform(padded, cv.DIST_L2, 5).max()))
        if radius >= 2:
            disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            blob = cv.morphologyEx(blob, cv.MORPH_OPEN, disc)
        trimmed[y0:y1, x0:x1] |= blob
    return trimmed


def _watershed_seeds(mask: np.ndarray) -> np.ndarray:
    """Sure-foreground seeds: the core of every blob, judged against its own size."""
    distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
    count, labels = cv.connectedComponents(mask)
    seeds = np.zeros_like(mask)
    for index in range(1, count):
        blob = labels == index
        seeds[blob & (distance >= SEED_FRACTION * distance[blob].max())] = 255
    return seeds


# ==========================================
# 1. ADVANCED OBJECT DETECTION
# (Integrated with Friend's Config Heuristics)
# ==========================================
def detect_objects(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")
    
    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_objects expects a BGR image with 3 channels.")
    
    image = np.ascontiguousarray(segmented_img, dtype=np.uint8)
    
    # 1. Candidate mask: foreground only, with thin attachments trimmed off
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    image_area = image.shape[0] * image.shape[1]
    candidate_mask = trim_thin_attachments(
        _candidate_mask(image), MIN_COMPONENT_FRACTION * image_area
    )

    # 2. Internal Watershed Separation
    sure_bg = cv.dilate(candidate_mask, kernel, iterations=3)

    valid_contours = []

    if candidate_mask.any():
        sure_fg = _watershed_seeds(candidate_mask)
        unknown = cv.subtract(sure_bg, sure_fg)

        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_image = image.copy()
        markers = cv.watershed(watershed_image, markers)
        
        for label in np.unique(markers):
            if label <= 1: 
                continue
            
            label_mask = np.uint8(markers == label) * 255
            contours, _ = cv.findContours(label_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv.contourArea)
                
                # Smooth the contour slightly to get rid of minor hand/leaf jaggedness
                epsilon = 0.005 * cv.arcLength(c, True)
                c = cv.approxPolyDP(c, epsilon, True)
                
                area = cv.contourArea(c)
                if not (MIN_AREA_FRACTION * image_area <= area <= MAX_AREA_FRACTION * image_area):
                    continue
                    
                if is_apple_shape(compute_shape_metrics(c)):
                    valid_contours.append(c)

    # --- NEW: relative-size filtering ---
    # Drop candidates that are much smaller than the largest candidate found.
    if valid_contours:
        areas = [cv.contourArea(c) for c in valid_contours]
        max_area = max(areas)
        # 0.25 threshold ensures we drop small leaf fragments while keeping real secondary apples
        size_keep = [c for c, a in zip(valid_contours, areas) if a >= 0.25 * max_area]
        valid_contours = size_keep

    # ❌ REMOVED THE MERGING LOOP THAT WAS EATING THE IMAGE ❌

    if not valid_contours:
        annotated_image = annotate_apples(image, [])
        return [], annotated_image, candidate_mask, "No apple found in the image"

    detected_apples = []

    for i, c in enumerate(valid_contours, start=1):
        x, y, w, h = cv.boundingRect(c)

        # We must also calculate confidence to prevent the UI from crashing
        # Re-calculating stats for the dictionary
        metrics = compute_shape_metrics(c)

        detected_apples.append({
            "id": i,
            "bbox": (x, y, w, h),
            "contour": c,
            "shape": metrics,
            # Pass a baseline confidence score based on solidity so app.py doesn't throw a KeyError
            "confidence": min(metrics["solidity"], 1.0),
            "individual_img": image.copy() 
        })

    annotated_image = annotate_apples(image, detected_apples)

    total_apples = len(valid_contours)
    status_message = f"Apple detected! Found {total_apples} apple(s)"

    return detected_apples, annotated_image, candidate_mask, status_message


def annotate_apples(image: np.ndarray, apples: list[dict]) -> np.ndarray:
    """Draw contour, box and "Apple N" label for each detection on a copy of ``image``."""
    annotated_image = image.copy()

    if not apples:
        cv.putText(annotated_image, "No apples found", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        return annotated_image

    for apple in apples:
        x, y, w, h = apple["bbox"]
        cv.drawContours(annotated_image, [apple["contour"]], -1, (0, 255, 0), 2)
        cv.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(annotated_image, f"Apple {apple['id']}", (x, max(20, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)

    return annotated_image


# ==========================================
# 2. EDGE DETECTION
# ==========================================
def detect_edges(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("detect_edges expects a BGR image with 3 channels.")

    segmented_img = np.ascontiguousarray(segmented_img, dtype=np.uint8)
    gray = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    smoothed = cv.bilateralFilter(normalized, d=7, sigmaColor=50, sigmaSpace=50)

    # Successful preprocessing now presents the segmented result on a white
    # background. Exclude that background when selecting adaptive thresholds.
    foreground_pixels = smoothed[gray < 250]
    if foreground_pixels.size == 0:
        foreground_pixels = smoothed.flatten()

    median_intensity = float(np.median(foreground_pixels))
    lower = int(max(20, 0.66 * median_intensity))
    upper = int(min(220, 1.33 * median_intensity))

    edges = cv.Canny(smoothed, lower, upper)
    
    foreground_mask = np.where(gray < 250, 255, 0).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel)
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel)

    edges = cv.bitwise_and(edges, edges, mask=foreground_mask)

    edge_overlay = segmented_img.copy()
    edge_overlay[edges > 0] = (0, 0, 255)

    return edges, edge_overlay


# ==========================================
# 3. WATERSHED SEGMENTATION (Comparison)
# ==========================================
def apply_watershed(segmented_img: np.ndarray):
    if segmented_img is None or segmented_img.size == 0:
        raise ValueError("Input image is empty.")

    if segmented_img.ndim != 3 or segmented_img.shape[2] != 3:
        raise ValueError("apply_watershed expects a BGR image with 3 channels.")

    segmented_img = np.ascontiguousarray(segmented_img, dtype=np.uint8)

    hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 40])
    upper_bound = np.array([180, 255, 255])
    binary = cv.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    
    _, sure_fg = cv.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    total_item = len(np.unique(markers)) - 1
    
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_image = segmented_img.copy()
    markers = cv.watershed(watershed_image, markers)
    watershed_image[markers == -1] = (0, 0, 255) 
    
    return watershed_image, markers, binary
