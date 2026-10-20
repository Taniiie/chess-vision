import sys
sys.path.append('.')
import cv2
import numpy as np
import os
from step6f_piece_type import get_warped_board, extract_piece_features, REAL_TEST_SQUARES

def diagnose_real_thresholds():
    warped = get_warped_board()
    square_size = warped.shape[0] // 8
    os.makedirs("scratch/thresh_debug", exist_ok=True)

    for sq, (r, c), true_label, _ in REAL_TEST_SQUARES:
        y1, y2 = r * square_size, (r + 1) * square_size
        x1, x2 = c * square_size, (c + 1) * square_size
        crop = warped[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Current thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Center crop threshold / GrabCut / adaptive mask
        # Since piece is at center, let's see centered thresholding:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        c_val = int(blur[50, 50])
        diff = cv2.absdiff(blur, c_val)
        _, thresh_center = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        cv2.imwrite(f"scratch/thresh_debug/{sq}_{true_label}_orig.jpg", crop)
        cv2.imwrite(f"scratch/thresh_debug/{sq}_{true_label}_thresh.jpg", thresh)
        cv2.imwrite(f"scratch/thresh_debug/{sq}_{true_label}_center_diff.jpg", thresh_center)

    print("Saved threshold debug images to scratch/thresh_debug/")

if __name__ == "__main__":
    diagnose_real_thresholds()
