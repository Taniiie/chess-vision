import sys
sys.path.append('.')
import cv2
import numpy as np
import os
from piece_color import warp_board

def find_corners(gray):
    found, corners = cv2.findChessboardCornersSB(gray, (7, 7), None)
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    return corners if found else None

def inspect_real_pieces():
    empty_img = cv2.imread("chessboard.jpg")
    if empty_img is None:
        empty_img = cv2.imread("chessboard1.jpg")
    if empty_img is None:
        print("Could not load empty chessboard photo")
        return

    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    corners = find_corners(empty_gray)
    if corners is None:
        print("Could not find inner corners in empty board photo")
        return

    img1 = cv2.imread("chessboard1_pieces.jpeg")
    if img1 is None:
        print("Could not load chessboard1_pieces.jpeg")
        return

    eh, ew = empty_img.shape[:2]
    oh, ow = img1.shape[:2]
    if (oh, ow) != (eh, ew):
        img1 = cv2.resize(img1, (ew, eh))

    warped1 = warp_board(img1, corners)
    os.makedirs("scratch/real_crops", exist_ok=True)
    cv2.imwrite("scratch/warped_board.jpg", warped1)

    labels_map = {
        (0, 0): ("rook", "a8_black_rook"),
        (0, 1): ("knight", "b8_black_knight"),
        (0, 2): ("bishop", "c8_black_bishop"),
        (0, 3): ("queen", "d8_black_queen"),
        (0, 4): ("king", "e8_black_king"),
        (0, 5): ("bishop", "f8_black_bishop"),
        (0, 6): ("knight", "g8_black_knight"),
        (0, 7): ("rook", "h8_black_rook"),
        (1, 0): ("pawn", "a7_black_pawn"),
        (1, 4): ("pawn", "e7_black_pawn"),
        (6, 3): ("pawn", "d2_white_pawn"),
        (6, 4): ("pawn", "e2_white_pawn"),
        (7, 0): ("rook", "a1_white_rook"),
        (7, 1): ("knight", "b1_white_knight"),
        (7, 2): ("bishop", "c1_white_bishop"),
        (7, 3): ("queen", "d1_white_queen"),
        (7, 4): ("king", "e1_white_king"),
        (7, 7): ("rook", "h1_white_rook"),
    }

    for (r, c), (ptype, name) in labels_map.items():
        y1, y2 = r * 100, (r + 1) * 100
        x1, x2 = c * 100, (c + 1) * 100
        crop = warped1[y1:y2, x1:x2]
        cv2.imwrite(f"scratch/real_crops/{name}.jpg", crop)
        print(f"Saved crop {name} ({ptype})")

if __name__ == "__main__":
    inspect_real_pieces()
