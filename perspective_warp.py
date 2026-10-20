"""
AI Chess Vision Coach - Step 6b: Perspective transform (flatten the board)

Takes the 49 INTERNAL corner points detected in Step 6a and:
1. Extrapolates the 4 OUTER corners of the actual board (the internal
   corners are one square inward from the true board edge on all sides).
2. Uses cv2.getPerspectiveTransform + cv2.warpPerspective to "unwarp"
   the board into a perfect top-down square, regardless of camera angle.

This is the same math used to digitally straighten a photographed
document - we're doing it to a chessboard instead.
"""

import cv2
import numpy as np

IMAGE_PATH = "chessboard1.jpg"  # your real board photo
CHESSBOARD_SIZE = (7, 7)       # internal corners for an 8x8 board
WARPED_SIZE = 800              # output will be an 800x800 top-down image


def extrapolate_outer_corners(corners, pattern_size):
    """
    corners: the (49, 1, 2) array from findChessboardCornersSB/Corners
    pattern_size: (cols, rows) of INTERNAL corners, e.g. (7, 7)

    Returns the 4 OUTER board corners (top-left, top-right,
    bottom-right, bottom-left) by extending one square-width beyond
    the extreme internal corners in each direction.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)

    top_left_in = grid[0, 0]
    top_right_in = grid[0, cols - 1]
    bottom_left_in = grid[rows - 1, 0]
    bottom_right_in = grid[rows - 1, cols - 1]

    # One "square step" along the top edge and left edge
    step_right = (top_right_in - top_left_in) / (cols - 1)
    step_down = (bottom_left_in - top_left_in) / (rows - 1)

    # Extrapolate each outer corner by stepping one more square
    # outward from the corresponding internal corner
    outer_tl = top_left_in - step_right - step_down
    outer_tr = top_right_in + step_right - step_down
    outer_bl = bottom_left_in - step_right + step_down
    outer_br = bottom_right_in + step_right + step_down

    return np.array([outer_tl, outer_tr, outer_br, outer_bl], dtype=np.float32)


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
        print("Board not detected - can't warp without corners.")
        return

    print("Board detected. Computing outer corners and warping...")

    # Draw the detected internal corners on the original image (for reference)
    reference = image.copy()
    cv2.drawChessboardCorners(reference, CHESSBOARD_SIZE, corners, found)

    outer_corners = extrapolate_outer_corners(corners.reshape(-1, 2), CHESSBOARD_SIZE)

    # Draw the extrapolated outer board boundary too, in a distinct color
    pts = outer_corners.astype(int)
    cv2.polylines(reference, [pts], isClosed=True, color=(255, 0, 255), thickness=3)

    # Destination: a perfect square, corners in the same order
    # (top-left, top-right, bottom-right, bottom-left)
    dst_points = np.array([
        [0, 0],
        [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1],
        [0, WARPED_SIZE - 1],
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(outer_corners, dst_points)
    warped = cv2.warpPerspective(image, transform_matrix, (WARPED_SIZE, WARPED_SIZE))

    # Draw the 8x8 grid lines on the warped result, as a visual sanity check
    warped_with_grid = warped.copy()
    step = WARPED_SIZE // 8
    for i in range(9):
        cv2.line(warped_with_grid, (i * step, 0), (i * step, WARPED_SIZE), (0, 255, 0), 1)
        cv2.line(warped_with_grid, (0, i * step), (WARPED_SIZE, i * step), (0, 255, 0), 1)

    # Resize the original/reference image for display if too large
    max_w = 700
    if reference.shape[1] > max_w:
        scale = max_w / reference.shape[1]
        reference = cv2.resize(reference, None, fx=scale, fy=scale)

    cv2.imshow("Original with detected corners + outer boundary", reference)
    cv2.imshow("Warped top-down view", warped)
    cv2.imshow("Warped with 8x8 grid overlay", warped_with_grid)

    print("Press any key (with an image window focused) to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
