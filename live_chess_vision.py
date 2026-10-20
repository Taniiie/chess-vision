"""
AI Chess Vision Coach - Live version: detection + warp + occupancy

Combines everything from Steps 6a-6d into one live pipeline:
- Live camera feed
- Board detection (with a status indicator)
- Press 'c' to CAPTURE the current frame and run full occupancy
  analysis (same variance-based logic as step6d)
- Press 'q' to quit

CAMERA SETUP: adjust CAMERA_INDEX and CAMERA_BACKEND below to match
whatever combination worked for you. If you're not sure, try:
  CAMERA_BACKEND = None       (let OpenCV auto-pick)
  CAMERA_BACKEND = cv2.CAP_DSHOW
  CAMERA_BACKEND = cv2.CAP_MSMF
"""

import cv2
import numpy as np

CAMERA_INDEX = 1
CAMERA_BACKEND = None

CHESSBOARD_SIZE = (7, 7)
WARPED_SIZE = 800
SQUARE_SIZE = WARPED_SIZE // 8


def open_camera():
    if CAMERA_BACKEND is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    return cap


def extrapolate_outer_corners(corners, pattern_size):
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    top_left_in = grid[0, 0]
    top_right_in = grid[0, cols - 1]
    bottom_left_in = grid[rows - 1, 0]
    bottom_right_in = grid[rows - 1, cols - 1]
    step_right = (top_right_in - top_left_in) / (cols - 1)
    step_down = (bottom_left_in - top_left_in) / (rows - 1)
    outer_tl = top_left_in - step_right - step_down
    outer_tr = top_right_in + step_right - step_down
    outer_bl = bottom_left_in - step_right + step_down
    outer_br = bottom_right_in + step_right + step_down
    return np.array([outer_tl, outer_tr, outer_br, outer_bl], dtype=np.float32)


def warp_board(image, corners):
    outer_corners = extrapolate_outer_corners(corners.reshape(-1, 2), CHESSBOARD_SIZE)
    dst_points = np.array([
        [0, 0], [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1], [0, WARPED_SIZE - 1],
    ], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(outer_corners, dst_points)
    return cv2.warpPerspective(image, transform_matrix, (WARPED_SIZE, WARPED_SIZE))


def get_square_crop(warped_gray, row, col, margin_ratio=0.2):
    y1 = row * SQUARE_SIZE
    x1 = col * SQUARE_SIZE
    margin = int(SQUARE_SIZE * margin_ratio)
    return warped_gray[y1 + margin: y1 + SQUARE_SIZE - margin,
                        x1 + margin: x1 + SQUARE_SIZE - margin]


def square_name(row, col):
    return f"{'abcdefgh'[col]}{8 - row}"


def expected_occupied(square):
    return int(square[1]) in (1, 2, 7, 8)


def analyze_occupancy(warped):
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    variances = {}
    for row in range(8):
        for col in range(8):
            crop = get_square_crop(warped_gray, row, col)
            variances[square_name(row, col)] = np.var(crop)

    threshold = sorted(variances.values())[31]

    labeled = warped.copy()
    correct = 0
    for row in range(8):
        for col in range(8):
            name = square_name(row, col)
            predicted = variances[name] > threshold
            actual = expected_occupied(name)
            correct += predicted == actual

            x1, y1 = col * SQUARE_SIZE, row * SQUARE_SIZE
            color = (0, 0, 255) if predicted else (0, 255, 0)
            thickness = 2 if predicted == actual else 4
            cv2.rectangle(labeled, (x1, y1), (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE), color, thickness)
            cv2.putText(labeled, name, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    print(f"Accuracy: {correct}/64 ({correct/64*100:.1f}%)  [assumes full starting position]")
    return labeled


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("Error: could not open camera. Try adjusting CAMERA_INDEX/CAMERA_BACKEND.")
        return

    print("PHASE 1: Point the camera at your EMPTY board.")
    print("Press 'e' to CALIBRATE (lock in the board corners) once it's detected.")
    print("Press 'q' to quit.")

    calibrated = False
    saved_corners = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)
        if not found:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                     cv2.CALIB_CB_NORMALIZE_IMAGE |
                     cv2.CALIB_CB_FILTER_QUADS)
            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        if not found:
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
            _, binary = cv2.threshold(smoothed, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            found, corners = cv2.findChessboardCornersSB(binary, CHESSBOARD_SIZE, None)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        display = frame.copy()

        if not calibrated:
            status = "BOARD DETECTED - press 'e' to calibrate" if found else "Searching for empty board..."
            status_color = (0, 255, 0) if found else (0, 0, 255)
            if found:
                cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        else:
            status = "CALIBRATED - place pieces, press 'c' to capture + analyze"
            status_color = (255, 255, 0)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Live Feed", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e') and not calibrated:
            if not found:
                print("No board detected right now - can't calibrate.")
                continue
            saved_corners = corners
            calibrated = True
            print("Calibrated! Now set up your pieces and press 'c' to capture + analyze.")
        elif key == ord('c') and calibrated:
            print("Captured! Analyzing using the SAVED calibration...")
            warped = warp_board(frame, saved_corners)
            result = analyze_occupancy(warped)
            cv2.imshow("Occupancy Result (press any key to close this window)", result)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()