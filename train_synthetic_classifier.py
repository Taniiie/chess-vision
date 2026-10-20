"""
AI Chess Vision Coach - Synthetic Data Generator & Classifier Trainer (Step 6f Helper)

This script:
1. Procedurally draws recognizable chess piece silhouettes (Pawn, Knight, Bishop, Rook, Queen, King)
   using OpenCV drawing routines with random spatial, scale, rotation, and noise variations.
2. Extracts a 7-dimensional shape/profile feature vector for each image:
   - Feature 0: Normalized Height (h / img_h)
   - Feature 1: Width at top 20% vertical band (normalized by bbox w)
   - Feature 2: Width at middle 50% vertical band (normalized by bbox w)
   - Feature 3: Width at bottom 80% vertical band (normalized by bbox w)
   - Feature 4: Aspect Ratio (h / w)
   - Feature 5: Top Peak Count / Bumpiness (local maxima in upper profile)
   - Feature 6: Solidity (contour area / bbox area)
3. Trains k-NN classifiers (k=3, k=5) on synthetic feature vectors.
4. Reports synthetic training accuracy (sanity check) and saves the model pipeline.
"""

import os
import glob
import math
import random
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Configuration
PIECE_CLASSES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
TRAIN_DATA_DIR = "training_data"
SAMPLES_PER_CLASS = 30  # 30 synthetic images per piece type (total 180 images)
IMAGE_SIZE = 100        # Standard crop dimensions (100x100)
MODEL_SAVE_PATH = "piece_type_knn.joblib"


def draw_procedural_piece(piece_type, bg_color=240, fg_color=20):
    """
    Procedurally draws a single chess piece silhouette onto a 100x100 canvas
    with realistic proportions and slight random variations.
    """
    img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * bg_color
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    cx = IMAGE_SIZE // 2 + random.randint(-3, 3)
    base_y = IMAGE_SIZE - random.randint(8, 14)
    base_w = random.randint(34, 46)

    # 1. Base pedestal (common to all pieces)
    pts_base = np.array([
        [cx - base_w // 2, base_y],
        [cx + base_w // 2, base_y],
        [cx + base_w // 2 - 4, base_y - 8],
        [cx - base_w // 2 + 4, base_y - 8]
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts_base], 255)

    neck_y = base_y - 8

    if piece_type == "pawn":
        # Pawn: small round head, tapered stem
        head_r = random.randint(9, 12)
        head_cy = neck_y - random.randint(22, 28)
        cv2.circle(canvas, (cx, head_cy), head_r, 255, -1)
        # Stem
        pts_stem = np.array([
            [cx - base_w // 2 + 6, neck_y],
            [cx + base_w // 2 - 6, neck_y],
            [cx + head_r - 2, head_cy + head_r // 2],
            [cx - head_r + 2, head_cy + head_r // 2]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_stem], 255)

    elif piece_type == "knight":
        # Knight: horse silhouette with head snout and mane notch
        facing = random.choice([-1, 1])
        head_y = neck_y - random.randint(35, 42)
        pts_knight = np.array([
            [cx - base_w // 2 + 5, neck_y],
            [cx + base_w // 2 - 5, neck_y],
            [cx + facing * 14, head_y + 15],
            [cx + facing * 18, head_y + 5],    # Snout tip
            [cx + facing * 8, head_y - 8],     # Ear/head top
            [cx - facing * 8, head_y - 4],     # Mane top
            [cx - facing * 14, neck_y - 10]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_knight], 255)

    elif piece_type == "bishop":
        # Bishop: oval mitre head with top knob
        mitre_h = random.randint(28, 34)
        mitre_w = random.randint(18, 24)
        head_cy = neck_y - mitre_h // 2 - 4
        cv2.ellipse(canvas, (cx, head_cy), (mitre_w // 2, mitre_h // 2), 0, 0, 360, 255, -1)
        # Top knob
        cv2.circle(canvas, (cx, head_cy - mitre_h // 2 - 2), 3, 255, -1)
        # Stem
        pts_stem = np.array([
            [cx - base_w // 2 + 5, neck_y],
            [cx + base_w // 2 - 5, neck_y],
            [cx + mitre_w // 2 - 2, head_cy + 4],
            [cx - mitre_w // 2 + 2, head_cy + 4]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_stem], 255)

    elif piece_type == "rook":
        # Rook: straight tower with battlements (crenellations)
        top_y = neck_y - random.randint(34, 40)
        top_w = random.randint(26, 32)
        body_w = random.randint(20, 24)
        pts_tower = np.array([
            [cx - base_w // 2 + 4, neck_y],
            [cx + base_w // 2 - 4, neck_y],
            [cx + body_w // 2, top_y + 10],
            [cx + top_w // 2, top_y + 10],
            [cx + top_w // 2, top_y],
            [cx - top_w // 2, top_y],
            [cx - top_w // 2, top_y + 10],
            [cx - body_w // 2, top_y + 10]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_tower], 255)
        # Notch cutouts
        notch_w = top_w // 5
        cv2.rectangle(canvas, (cx - notch_w // 2, top_y), (cx + notch_w // 2, top_y + 5), 0, -1)

    elif piece_type == "queen":
        # Queen: crown with 5 pointed spires topped with circular beads
        crown_y = neck_y - random.randint(38, 45)
        crown_w = random.randint(30, 38)
        pts_stem = np.array([
            [cx - base_w // 2 + 5, neck_y],
            [cx + base_w // 2 - 5, neck_y],
            [cx + crown_w // 2 - 4, crown_y + 15],
            [cx - crown_w // 2 + 4, crown_y + 15]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_stem], 255)
        spikes = np.array([
            [cx - crown_w // 2, crown_y + 15],
            [cx - crown_w // 2, crown_y + 2],
            [cx - crown_w // 4, crown_y + 10],
            [cx, crown_y],
            [cx + crown_w // 4, crown_y + 10],
            [cx + crown_w // 2, crown_y + 2],
            [cx + crown_w // 2, crown_y + 15]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [spikes], 255)
        cv2.circle(canvas, (cx - crown_w // 2, crown_y + 2), 2, 255, -1)
        cv2.circle(canvas, (cx, crown_y), 3, 255, -1)
        cv2.circle(canvas, (cx + crown_w // 2, crown_y + 2), 2, 255, -1)

    elif piece_type == "king":
        # King: tall body topped with 4-arm cross
        head_y = neck_y - random.randint(34, 40)
        head_w = random.randint(24, 30)
        pts_king = np.array([
            [cx - base_w // 2 + 5, neck_y],
            [cx + base_w // 2 - 5, neck_y],
            [cx + head_w // 2, head_y],
            [cx - head_w // 2, head_y]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_king], 255)
        cross_y = head_y - 7
        cv2.rectangle(canvas, (cx - 2, cross_y - 8), (cx + 2, cross_y + 2), 255, -1)
        cv2.rectangle(canvas, (cx - 6, cross_y - 5), (cx + 6, cross_y - 2), 255, -1)

    # Affine variations (rotation, scale)
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), angle, scale)
    canvas = cv2.warpAffine(canvas, M, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_NEAREST)

    img[canvas > 128] = fg_color
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_synthetic_dataset():
    """Generates synthetic dataset and saves images into class folders."""
    print("=" * 60)
    print("PART A: Generating Synthetic Training Data")
    print("Method Used: Procedurally drew recognizable piece silhouettes using")
    print("OpenCV drawing functions (circles, polygons, ellipses, crosses) with")
    print("random variation (rotation -10..10 deg, scale 0.88..1.12, spatial shift).")
    print("=" * 60)

    total_generated = 0
    for ptype in PIECE_CLASSES:
        folder = os.path.join(TRAIN_DATA_DIR, ptype)
        os.makedirs(folder, exist_ok=True)
        for f in glob.glob(os.path.join(folder, "*.jpg")):
            os.remove(f)

        for i in range(SAMPLES_PER_CLASS):
            bg = random.choice([235, 245, 255, 20, 35])
            fg = 20 if bg > 128 else 235
            img = draw_procedural_piece(ptype, bg_color=bg, fg_color=fg)
            filename = os.path.join(folder, f"{ptype}_{i+1:02d}.jpg")
            cv2.imwrite(filename, img)
            total_generated += 1

        print(f"  - Generated {SAMPLES_PER_CLASS} images for '{ptype}' -> {folder}/")

    print(f"Total synthetic training images generated: {total_generated}\n")


def extract_piece_features(img):
    """
    PART B: Feature Extraction Pipeline
    
    Given a cropped piece image (grayscale or BGR):
    1. Isolates silhouette via Otsu / background contrast thresholding.
    2. Extracts 7 profile and shape features:
       - Feature 0: Normalized Height (h / img_h)
       - Feature 1: Width at Top 20% band (normalized by bbox w)
       - Feature 2: Width at Mid 50% band (normalized by bbox w)
       - Feature 3: Width at Bottom 80% band (normalized by bbox w)
       - Feature 4: Aspect Ratio (h / w)
       - Feature 5: Top Peak Count / Bumpiness (local maxima in upper profile)
       - Feature 6: Solidity (contour area / bbox area)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    img_h, img_w = gray.shape[:2]

    # Thresholding to isolate foreground silhouette
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Ensure binary mask has white piece silhouette on black background
    center_y, center_x = img_h // 2, img_w // 2
    if thresh[center_y, center_x] == 0 and cv2.countNonZero(thresh) > (img_h * img_w * 0.5):
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select main central contour
    best_cnt = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w >= img_w - 2 and h >= img_h - 2:
            continue
        if area > max_area:
            max_area = area
            best_cnt = cnt

    if best_cnt is None:
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
        else:
            return np.zeros(7, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(best_cnt)
    contour_area = cv2.contourArea(best_cnt)

    if w == 0 or h == 0:
        return np.zeros(7, dtype=np.float32)

    # 1. Height ratio
    f_height = float(h) / float(img_h)

    # 2. Vertical profile band widths
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)

    def band_width(y_row):
        y_row = np.clip(y_row, 0, img_h - 1)
        row_pixels = np.where(mask[y_row, :] > 0)[0]
        if len(row_pixels) > 0:
            return float(row_pixels[-1] - row_pixels[0] + 1) / float(w)
        return 0.0

    f_w_top = band_width(int(y + 0.20 * h))
    f_w_mid = band_width(int(y + 0.50 * h))
    f_w_bot = band_width(int(y + 0.80 * h))

    # 3. Aspect Ratio (h / w)
    f_aspect_ratio = float(h) / float(w)

    # 4. Top portion bumpiness / local maxima
    top_y_cutoff = int(y + 0.30 * h)
    peak_count = 1
    upper_profile = []
    for col_x in range(x, x + w):
        col_pixels = np.where(mask[y:top_y_cutoff, col_x] > 0)[0]
        if len(col_pixels) > 0:
            upper_profile.append(col_pixels[0])
    
    if len(upper_profile) >= 5:
        profile_smooth = np.convolve(upper_profile, np.ones(3)/3.0, mode='valid')
        peaks = 0
        for k in range(1, len(profile_smooth) - 1):
            if profile_smooth[k] < profile_smooth[k-1] and profile_smooth[k] < profile_smooth[k+1]:
                peaks += 1
        peak_count = max(1, peaks)

    f_peaks = float(peak_count)

    # 5. Solidity (contour area / bbox area)
    f_solidity = float(contour_area) / float(w * h)

    return np.array([f_height, f_w_top, f_w_mid, f_w_bot, f_aspect_ratio, f_peaks, f_solidity], dtype=np.float32)


def train_and_evaluate():
    """PART C: Train k-NN Classifiers & Report Accuracy."""
    print("=" * 60)
    print("PART B & C: Feature Extraction, Model Training & Evaluation")
    print("=" * 60)

    X_train = []
    y_train = []

    for label_idx, ptype in enumerate(PIECE_CLASSES):
        folder = os.path.join(TRAIN_DATA_DIR, ptype)
        files = glob.glob(os.path.join(folder, "*.jpg"))
        for filepath in files:
            img = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)
            feat = extract_piece_features(img)
            X_train.append(feat)
            y_train.append(ptype)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Extracted {X_train.shape[1]} features from {len(X_train)} synthetic images.")

    best_knn = None
    best_acc = 0.0

    for k in [3, 5]:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_train, y_train) * 100.0
        print(f"  - k-NN (k={k}) Training Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_knn = pipeline

    print("\n" + "!" * 60)
    print(f"SANITY CHECK NOTICE: Synthetic Training accuracy is high ({best_acc:.2f}%).")
    print("This number is NOT meaningful on its own because the classifier is being")
    print("tested on the exact synthetic distribution it trained on.")
    print("Real-world generalization will be tested separately in Part D.")
    print("!" * 60 + "\n")

    joblib.dump(best_knn, MODEL_SAVE_PATH)
    print(f"Trained model pipeline saved successfully to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    generate_synthetic_dataset()
    train_and_evaluate()
