"""
AI Chess Vision Coach - Step 6a (static image version)

Instead of a live webcam feed, this reads a single photo of your
chessboard and tries to detect it. Much simpler to test with, and
avoids all webcam driver headaches.

HOW TO USE:
1. Take a photo of your EMPTY chessboard with your phone's camera app
   (no pieces on it - the detector needs to see the checkered pattern).
2. Transfer that photo to your laptop and put it in this project folder.
3. Set IMAGE_PATH below to match the photo's filename.
4. Run this script.
"""

import cv2

IMAGE_PATH = "chessboard1.jpg"       # use the high-contrast board; change to your photo
CHESSBOARD_SIZE = (7, 7)  # internal corners for an 8x8 board


def detect_board_robust(gray, pattern_size):
    """
    Tries multiple detection strategies, from fast to slow, to handle
    low-contrast wooden boards, uneven lighting, etc.
    Returns (found, corners).
    """
    # Strategy 1: SB detector (fast, good for clean boards)
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if found:
        return True, corners

    # Strategy 2: Classic detector with enhanced flags
    import numpy as np
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FILTER_QUADS)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found:
        return True, corners

    # Strategy 3: CLAHE contrast enhancement + bilateral filter + threshold
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _, binary = cv2.threshold(smoothed, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    found, corners = cv2.findChessboardCornersSB(binary, pattern_size, None)
    if found:
        return True, corners

    # Strategy 4: plain classic on original (last resort)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    return found, corners

# Load the image from disk
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: could not load '{IMAGE_PATH}'.")
    print("Check that the file exists in this folder and the filename/extension matches exactly.")
    exit()

print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# findChessboardCornersSB is a newer, more robust detector than the
# classic findChessboardCorners - it handles real-world lighting and
# lower-contrast patterns (like wood-tone boards) much better.
found, corners = detect_board_robust(gray, CHESSBOARD_SIZE)

if found:
    print("Board DETECTED!")
    cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners, found)
    cv2.putText(
        image, "BOARD DETECTED", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
    )
else:
    print("Board NOT detected. Try a clearer/more top-down photo with better lighting.")
    cv2.putText(
        image, "Board not detected", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
    )

# Resize for display if the photo is very large (phone photos often are)
max_display_width = 900
if image.shape[1] > max_display_width:
    scale = max_display_width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale)

cv2.imshow("Board Detection Result", image)
print("Press any key (with the image window focused) to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()