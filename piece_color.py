"""
AI Chess Vision Coach - Step 6e: Piece Color Classification (White vs Black)

Core Idea:
After detecting which squares are OCCUPIED (Step 6d), we need to determine
if the piece sitting on each occupied square is WHITE or BLACK.

Why sample a SMALL CENTER CIRCLE instead of the whole square?
- A chess square contains both the PIECE and the BOARD BACKGROUND (wood/color).
- The board background can be light or dark, which would skew the average brightness.
- The body/head of the piece sits reliably at the exact CENTER of the square.
- By sampling a small circle at the center (radius ~15% of square width), we measure
  the piece's material color while ignoring most of the square background.

Auto-calibration using Ground Truth:
- In a starting position photo, Ranks 1-2 are ALWAYS White, Ranks 7-8 are ALWAYS Black.
- We measure brightness for known White pieces and known Black pieces.
- Threshold = midpoint between the lowest White brightness and highest Black brightness.
"""

import cv2
import numpy as np

# Image configuration (matches Step 6d)
EMPTY_IMAGE_PATH = "chessboard.jpg"          
OCCUPIED_IMAGE_PATH = "chessboard1_pieces.jpeg" 
CHESSBOARD_SIZE = (7, 7)
WARPED_SIZE = 800
SQUARE_SIZE = WARPED_SIZE // 8  # 100x100 pixels per square


def extrapolate_outer_corners(corners, pattern_size):
    """
    Extrapolates the 4 outer corners of the board from internal corners grid.
    """
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
    """
    Warps the detected board rectangle into a top-down WARPED_SIZE x WARPED_SIZE image.
    """
    outer_corners = extrapolate_outer_corners(corners.reshape(-1, 2), CHESSBOARD_SIZE)
    dst_points = np.array([
        [0, 0], [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1], [0, WARPED_SIZE - 1],
    ], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(outer_corners, dst_points)
    return cv2.warpPerspective(image, transform_matrix, (WARPED_SIZE, WARPED_SIZE))


def square_name(row, col):
    """Converts (row, col) matrix indices to chess square notation (e.g. e4)."""
    file_letter = "abcdefgh"[col]
    rank_number = 8 - row
    return f"{file_letter}{rank_number}"


def expected_occupied(square):
    """Ground truth occupancy for full starting position: ranks 1, 2, 7, 8."""
    rank = int(square[1])
    return rank in (1, 2, 7, 8)


def expected_color(square):
    """
    Ground truth color for full starting position:
    - Ranks 1 and 2: White pieces
    - Ranks 7 and 8: Black pieces
    """
    rank = int(square[1])
    if rank in (1, 2):
        return "WHITE"
    elif rank in (7, 8):
        return "BLACK"
    return None


def get_center_circular_brightness(warped_gray, row, col, radius_ratio=0.15):
    """
    Samples a small circular region at the exact center of a square
    and calculates its average grayscale brightness (0-255).

    Why a circle at center?
    - Prevents board square colors (light vs dark square) from corrupting piece color.
    - Captures the primary surface of the piece head/body.
    """
    cx = col * SQUARE_SIZE + SQUARE_SIZE // 2
    cy = row * SQUARE_SIZE + SQUARE_SIZE // 2
    radius = int(SQUARE_SIZE * radius_ratio)

    # Create a blank mask of the same size as the warped image
    mask = np.zeros_like(warped_gray, dtype=np.uint8)
    
    # Draw a solid white circle at the center of the square
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Compute mean grayscale intensity inside the mask
    mean_brightness = cv2.mean(warped_gray, mask=mask)[0]
    return mean_brightness, (cx, cy, radius)


def main():
    # STEP 1: Load empty board image for calibration
    empty_image = cv2.imread(EMPTY_IMAGE_PATH)
    if empty_image is None:
        print(f"Error: could not load '{EMPTY_IMAGE_PATH}' (empty board photo).")
        return

    empty_gray = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)

    # Multi-strategy detection for robust board localization
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

    print("Calibrated board corners successfully.")

    # STEP 2: Load occupied board photo and warp it
    occupied_image = cv2.imread(OCCUPIED_IMAGE_PATH)
    if occupied_image is None:
        print(f"Error: could not load '{OCCUPIED_IMAGE_PATH}' (occupied board photo).")
        return

    empty_h, empty_w = empty_image.shape[:2]
    occupied_h, occupied_w = occupied_image.shape[:2]
    if (occupied_h, occupied_w) != (empty_h, empty_w):
        print(f"Resizing occupied photo ({occupied_w}x{occupied_h}) to match "
              f"empty photo ({empty_w}x{empty_h})...")
        occupied_image = cv2.resize(occupied_image, (empty_w, empty_h))

    warped = warp_board(occupied_image, corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # STEP 3: Sample center brightness for all occupied squares
    white_brightnesses = []
    black_brightnesses = []
    square_brightness = {}
    square_circles = {}

    for row in range(8):
        for col in range(8):
            name = square_name(row, col)
            if expected_occupied(name):  # Occupied squares in starting position
                brightness, circle_info = get_center_circular_brightness(warped_gray, row, col)
                square_brightness[name] = brightness
                square_circles[name] = circle_info

                ground_truth = expected_color(name)
                if ground_truth == "WHITE":
                    white_brightnesses.append(brightness)
                elif ground_truth == "BLACK":
                    black_brightnesses.append(brightness)

    # STEP 4: Compute threshold & check for range overlap
    min_white = min(white_brightnesses)
    max_white = max(white_brightnesses)
    min_black = min(black_brightnesses)
    max_black = max(black_brightnesses)

    # Midpoint threshold between lowest known-white and highest known-black
    threshold = (min_white + max_black) / 2.0

    print("\n--- Color Calibration Stats ---")
    print(f"White pieces brightness range: {min_white:.1f} to {max_white:.1f} (avg: {np.mean(white_brightnesses):.1f})")
    print(f"Black pieces brightness range: {min_black:.1f} to {max_black:.1f} (avg: {np.mean(black_brightnesses):.1f})")
    print(f"Auto-computed White/Black Threshold: {threshold:.1f}")

    if min_white < max_black:
        print(f"[WARNING] White and Black brightness ranges OVERLAP! "
              f"(Lowest White = {min_white:.1f} < Highest Black = {max_black:.1f})")
        print("  -> Lighting/contrast or grid alignment needs improvement for 100% separation.")
    else:
        print("[SUCCESS] White and Black brightness ranges are cleanly separated!")

    # STEP 5: Classify each occupied square & evaluate accuracy
    correct = 0
    total_occupied = 32
    wrong_squares = []
    labeled = warped.copy()

    for row in range(8):
        for col in range(8):
            name = square_name(row, col)
            if not expected_occupied(name):
                continue  # Skip empty squares for color test

            brightness = square_brightness[name]
            predicted_color = "WHITE" if brightness > threshold else "BLACK"
            actual_color = expected_color(name)
            is_correct = (predicted_color == actual_color)

            if is_correct:
                correct += 1
            else:
                wrong_squares.append(f"{name} (pred: {predicted_color}, act: {actual_color}, bright: {brightness:.1f})")

            # Visual drawing
            x1, y1 = col * SQUARE_SIZE, row * SQUARE_SIZE
            # Cyan for White prediction, Magenta for Black prediction
            color = (255, 255, 0) if predicted_color == "WHITE" else (255, 0, 255)
            thickness = 2 if is_correct else 4  # Thick border = wrong color prediction

            # Draw square border
            cv2.rectangle(labeled, (x1, y1), (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE), color, thickness)

            # Draw sampled center circle
            cx, cy, r = square_circles[name]
            cv2.circle(labeled, (cx, cy), r, (0, 255, 0), 1)

            # Text label: Square name + W/B prediction
            label_text = f"{name}:{predicted_color[0]}"
            cv2.putText(labeled, label_text, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    accuracy = (correct / total_occupied) * 100
    print(f"\n--- Accuracy Report ---")
    print(f"Piece Color Accuracy: {correct}/{total_occupied} correct ({accuracy:.1f}%)")

    if wrong_squares:
        print("Misclassified squares (thick border):")
        for ws in wrong_squares:
            print(f"  - {ws}")
    else:
        print("All occupied squares correctly classified by color!")

    # Save output visualization image
    cv2.imwrite("piece_color_result.jpg", labeled)
    print("\nSaved labeled visualization to 'piece_color_result.jpg'.")

    try:
        cv2.imshow("Piece Color Classification (Cyan=White, Magenta=Black, Green Circle=Sampled Region)", labeled)
        print("Press any key (with image window focused) to close.")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
