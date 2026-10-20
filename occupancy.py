"""
AI Chess Vision Coach - Step 6d: Empty vs Occupied classification

Core idea: an EMPTY square is fairly uniform (just plain wood/board
color). A square with a PIECE on it has much more visual complexity -
edges, shadows, curves, highlights. We measure this "complexity" using
variance: how much do pixel values differ from each other within the
square?

Low variance -> probably empty
High variance -> probably occupied

We test this against a FULL STARTING POSITION, where we know exactly
which 32 squares should be occupied (ranks 1,2,7,8) and which 32
should be empty (ranks 3,4,5,6) - so we can auto-check accuracy.
"""

import cv2
import numpy as np

EMPTY_IMAGE_PATH = "chessboard.jpg"          
OCCUPIED_IMAGE_PATH = "chessboard1_pieces.jpeg" 
CHESSBOARD_SIZE = (7, 7)
WARPED_SIZE = 800
SQUARE_SIZE = WARPED_SIZE // 8


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
    """
    Crops a square, but shrinks inward by margin_ratio on each side.
    This avoids the grid LINE itself (the edge between two squares)
    from being counted as "complexity" - we only want to measure the
    complexity of what's actually inside the square.
    """
    y1 = row * SQUARE_SIZE
    x1 = col * SQUARE_SIZE
    margin = int(SQUARE_SIZE * margin_ratio)
    return warped_gray[y1 + margin: y1 + SQUARE_SIZE - margin,
                        x1 + margin: x1 + SQUARE_SIZE - margin]


def square_name(row, col):
    file_letter = "abcdefgh"[col]
    rank_number = 8 - row
    return f"{file_letter}{rank_number}"


def expected_occupied(square):
    """Ground truth for a full starting position: ranks 1,2,7,8 occupied."""
    rank = int(square[1])
    return rank in (1, 2, 7, 8)


def main():
    # STEP A: Calibrate using the EMPTY board photo
    empty_image = cv2.imread(EMPTY_IMAGE_PATH)
    if empty_image is None:
        print(f"Error: could not load '{EMPTY_IMAGE_PATH}' (empty board photo).")
        return

    empty_gray = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)

    # Multi-strategy detection for robustness with real-world boards
    found, corners = cv2.findChessboardCornersSB(empty_gray, CHESSBOARD_SIZE, None)
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(empty_gray, CHESSBOARD_SIZE, flags)
    if not found:
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(empty_gray)
        smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
        _, binary = cv2.threshold(smoothed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        found, corners = cv2.findChessboardCornersSB(binary, CHESSBOARD_SIZE, None)
    if not found:
        found, corners = cv2.findChessboardCorners(empty_gray, CHESSBOARD_SIZE, None)
    if not found:
        print("Could not detect the board in the EMPTY photo - can't calibrate.")
        return

    print("Calibrated successfully using the empty board photo.")

    # STEP B: Load the OCCUPIED board photo and warp it using the SAME calibration
    occupied_image = cv2.imread(OCCUPIED_IMAGE_PATH)
    if occupied_image is None:
        print(f"Error: could not load '{OCCUPIED_IMAGE_PATH}' (occupied board photo).")
        return

    # IMPORTANT: the transform was calculated using pixel coordinates
    # from the EMPTY photo's dimensions. If the occupied photo is a
    # different resolution, we must resize it to match first, or the
    # transform will map to completely wrong pixel locations.
    empty_h, empty_w = empty_image.shape[:2]
    occupied_h, occupied_w = occupied_image.shape[:2]
    if (occupied_h, occupied_w) != (empty_h, empty_w):
        print(f"Resizing occupied photo ({occupied_w}x{occupied_h}) to match "
              f"empty photo ({empty_w}x{empty_h})...")
        occupied_image = cv2.resize(occupied_image, (empty_w, empty_h))

    warped = warp_board(occupied_image, corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Step 1: compute variance for every square
    variances = {}
    for row in range(8):
        for col in range(8):
            crop = get_square_crop(warped_gray, row, col)
            variance = np.var(crop)
            variances[square_name(row, col)] = variance

    # Step 2: pick a threshold. Since we KNOW exactly 32 squares should
    # be occupied in a full starting position, we use the median
    # variance as the threshold - this adapts automatically to your
    # specific lighting/board instead of a hardcoded magic number.
    sorted_variances = sorted(variances.values())
    threshold = sorted_variances[31]  # the boundary between the 32 lowest and 32 highest
    print(f"Auto-computed threshold: {threshold:.1f}")

    # Step 3: classify + check against ground truth
    correct = 0
    labeled = warped.copy()

    for row in range(8):
        for col in range(8):
            name = square_name(row, col)
            predicted_occupied = variances[name] > threshold
            actual_occupied = expected_occupied(name)
            is_correct = predicted_occupied == actual_occupied
            correct += is_correct

            x1, y1 = col * SQUARE_SIZE, row * SQUARE_SIZE
            color = (0, 0, 255) if predicted_occupied else (0, 255, 0)  # red=occupied, green=empty
            thickness = 2 if is_correct else 4  # thick border = misclassified, easy to spot
            cv2.rectangle(labeled, (x1, y1), (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE), color, thickness)
            cv2.putText(
                labeled, name, (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
            )

    accuracy = correct / 64 * 100
    print(f"Accuracy: {correct}/64 correct ({accuracy:.1f}%)")

    wrong_squares = []
    for row in range(8):
        for col in range(8):
            name = square_name(row, col)
            if (variances[name] > threshold) != expected_occupied(name):
                wrong_squares.append(name)
    if wrong_squares:
        print("Misclassified squares (thick border):", ", ".join(wrong_squares))
    else:
        print("No misclassified squares! All 64 squares correct.")

    cv2.imwrite("occupancy_result.jpg", labeled)
    print("Saved labeled visualization to 'occupancy_result.jpg'.")

    try:
        cv2.imshow("Occupancy detection (green=empty, red=occupied, thick=wrong)", labeled)
        print("Press any key (with image window focused) to close.")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()