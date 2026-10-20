import sys
sys.path.append('.')
import cv2
import numpy as np

def extract_silhouette_robust(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h_img, w_img = gray.shape[:2]

    # Sample border background (outer 5 pixels border)
    border_mask = np.ones_like(gray, dtype=np.uint8) * 255
    border_mask[6:h_img-6, 6:w_img-6] = 0
    bg_val = float(np.median(gray[border_mask > 0]))

    # Absolute difference from board background
    diff = cv2.absdiff(gray.astype(np.float32), bg_val).astype(np.uint8)
    blur_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold difference
    _, thresh = cv2.threshold(blur_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours near center of crop
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = w_img // 2, h_img // 2

    valid_cnts = []
    for cnt in contours:
        # Check if contour is not the entire image border and intersects central zone
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= w_img - 4 and h >= h_img - 4:
            continue
        dist_to_center = cv2.pointPolygonTest(cnt, (cx, cy), True)
        # Or distance from bbox center to image center
        bbox_cx, bbox_cy = x + w / 2, y + h / 2
        d_center = np.hypot(bbox_cx - cx, bbox_cy - cy)
        if dist_to_center >= -15 or d_center < 30:
            valid_cnts.append((cv2.contourArea(cnt), cnt))

    if valid_cnts:
        valid_cnts.sort(key=lambda x: x[0], reverse=True)
        best_cnt = valid_cnts[0][1]
    elif contours:
        best_cnt = max(contours, key=cv2.contourArea)
    else:
        best_cnt = None

    return thresh, best_cnt

def test_robust_threshold():
    from step6f_piece_type import get_warped_board, REAL_TEST_SQUARES
    warped = get_warped_board()
    sq_size = warped.shape[0] // 8
    
    for sq, (r, c), true_label, _ in REAL_TEST_SQUARES:
        crop = warped[r*sq_size:(r+1)*sq_size, c*sq_size:(c+1)*sq_size]
        thresh, cnt = extract_silhouette_robust(crop)
        if cnt is not None:
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"Square {sq} ({true_label}): bbox=({x},{y},{w},{h}), area={cv2.contourArea(cnt):.0f}")
        else:
            print(f"Square {sq} ({true_label}): NO CONTOUR FOUND")

if __name__ == "__main__":
    test_robust_threshold()
