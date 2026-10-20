"""
AI Chess Vision Coach - Step 6c: Split the warped board into 64 squares

Builds on Step 6b. Since the warped image is a perfect WARPED_SIZE x
WARPED_SIZE square, each of the 64 chess squares is just a fixed-size
region we can slice out with simple math - no more corner detection
needed at this stage.

We also map each square to its real chess notation (a1-h8), which
we'll need later to build the FEN string.
"""

import cv2
import numpy as np

IMAGE_PATH = "chessboard1.jpg"
CHESSBOARD_SIZE = (7, 7)
WARPED_SIZE = 800
SQUARE_SIZE = WARPED_SIZE // 8  # 100x100 pixels per square


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
        [0, 0],
        [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1],
        [0, WARPED_SIZE - 1],
    ], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(outer_corners, dst_points)
    return cv2.warpPerspective(image, transform_matrix, (WARPED_SIZE, WARPED_SIZE))


def split_into_squares(warped_image):
    """
    Slices the warped board into 64 squares and labels each with its
    chess notation (a1-h8).

    IMPORTANT ASSUMPTION: this assumes the board was photographed with
    White's side at the BOTTOM of the image (standard viewing angle).
    Row 0 (top of image) = rank 8, row 7 (bottom of image) = rank 1.
    Column 0 (left) = file a, column 7 (right) = file h.
    If your photo has the board rotated differently, the labels will
    be wrong even though the slicing itself is correct - we'll handle
    this properly in a later step.
    """
    squares = {}
    files = "abcdefgh"

    for row in range(8):
        for col in range(8):
            y1 = row * SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            x1 = col * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE

            square_image = warped_image[y1:y2, x1:x2]

            file_letter = files[col]
            rank_number = 8 - row  # row 0 = rank 8, row 7 = rank 1
            square_name = f"{file_letter}{rank_number}"

            squares[square_name] = square_image

    return squares


def draw_labeled_grid(warped_image, squares):
    """Draws the grid lines and square names for a visual sanity check."""
    labeled = warped_image.copy()
    for row in range(8):
        for col in range(8):
            x1 = col * SQUARE_SIZE
            y1 = row * SQUARE_SIZE
            file_letter = "abcdefgh"[col]
            rank_number = 8 - row
            label = f"{file_letter}{rank_number}"

            cv2.rectangle(
                labeled, (x1, y1), (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE),
                (0, 255, 0), 1
            )
            cv2.putText(
                labeled, label, (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
    return labeled


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: could not load '{IMAGE_PATH}'.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Multi-strategy detection for robustness with real-world boards
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
    if not found:
        print("Board not detected.")
        return

    warped = warp_board(image, corners)
    squares = split_into_squares(warped)

    print(f"Split into {len(squares)} squares.")
    print("Example square names:", list(squares.keys())[:8], "...")

    labeled = draw_labeled_grid(warped, squares)
    cv2.imshow("Warped board with square labels (a1-h8)", labeled)

    # Show a couple of individual squares as a demonstration
    cv2.imshow("Example: square e4", squares["e4"])
    cv2.imshow("Example: square a1", squares["a1"])

    print("Press any key (with an image window focused) to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
