"""
Quick verification: test all non-GUI components of ChessVisionCoach.
"""

import cv2
import numpy as np
import chess
import chess.engine
import os
import sys

STOCKFISH_PATH = r"stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


# ---------- TEST 1: Stockfish Engine ----------
print("=== TEST 1: Stockfish Engine ===")
try:
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    info = engine.analyse(board, chess.engine.Limit(time=0.3))
    score = info["score"]
    best = engine.play(board, chess.engine.Limit(time=0.3)).move
    engine.quit()
    print(f"  Evaluation: {score}, Best move: {best}")
    test("Engine connects and evaluates", True)
except Exception as e:
    test(f"Engine connects and evaluates ({e})", False)

# ---------- TEST 2: File existence checks ----------
print("\n=== TEST 2: Required Files ===")
test("chessboard1.jpg exists", os.path.exists("chessboard1.jpg"))
test("chesspieces.jpeg exists", os.path.exists("chesspieces.jpeg"))
test("Stockfish exe exists", os.path.exists(STOCKFISH_PATH))
test("assets/pieces dir exists", os.path.exists("assets/pieces"))
expected_pieces = ["wP.png", "wN.png", "wB.png", "wR.png", "wQ.png", "wK.png",
                   "bP.png", "bN.png", "bB.png", "bR.png", "bQ.png", "bK.png"]
for p in expected_pieces:
    test(f"  Piece image {p}", os.path.exists(os.path.join("assets", "pieces", p)))

# ---------- TEST 3: check_file_type ----------
print("\n=== TEST 3: check_file_type logic ===")
try:
    with open("chessboard1.jpg", "rb") as f:
        header = f.read(12)
    is_jpeg = header.startswith(b'\xff\xd8\xff')
    print(f"  Header: {header.hex()}")
    test("chessboard1.jpg is a valid JPEG", is_jpeg)
except Exception as e:
    test(f"File type check ({e})", False)

# ---------- TEST 4: Board detection ----------
print("\n=== TEST 4: Board Detection (chessboard1.jpg) ===")
try:
    img = cv2.imread("chessboard1.jpg")
    test("Image loaded", img is not None)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(gray, (7, 7), None)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    test("7x7 corners detected", found)
    if found:
        test("Correct corner count (49)", corners.shape[0] == 49)
except Exception as e:
    test(f"Board detection ({e})", False)

# ---------- TEST 5: Perspective Warp ----------
print("\n=== TEST 5: Perspective Warp ===")
try:
    def extrapolate_outer_corners(corners, pattern_size):
        cols, rows = pattern_size
        grid = corners.reshape(rows, cols, 2)
        tl = grid[0, 0]; tr = grid[0, cols - 1]
        bl = grid[rows - 1, 0]; br = grid[rows - 1, cols - 1]
        sr = (tr - tl) / (cols - 1)
        sd = (bl - tl) / (rows - 1)
        return np.array([tl - sr - sd, tr + sr - sd,
                         br + sr + sd, bl - sr + sd], dtype=np.float32)

    outer = extrapolate_outer_corners(corners.reshape(-1, 2), (7, 7))
    test("Outer corners extrapolated (4 points)", outer.shape == (4, 2))

    dst = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(outer, dst)
    warped = cv2.warpPerspective(img, M, (800, 800))
    test("Warped to 800x800", warped.shape == (800, 800, 3))
except Exception as e:
    test(f"Perspective warp ({e})", False)

# ---------- TEST 6: Split into 64 squares ----------
print("\n=== TEST 6: Split into 64 Squares ===")
try:
    SQUARE_SIZE = 100
    squares = {}
    for row in range(8):
        for col in range(8):
            y1 = row * SQUARE_SIZE
            x1 = col * SQUARE_SIZE
            sq = warped[y1:y1 + SQUARE_SIZE, x1:x1 + SQUARE_SIZE]
            name = f"{'abcdefgh'[col]}{8 - row}"
            squares[name] = sq
    test("Split into 64 squares", len(squares) == 64)
    test("Square e4 shape is 100x100", squares["e4"].shape == (100, 100, 3))
except Exception as e:
    test(f"Square splitting ({e})", False)

# ---------- TEST 7: Chess GUI imports ----------
print("\n=== TEST 7: Chess GUI Imports ===")
try:
    import pygame
    test("pygame importable", True)
    test("python-chess importable", True)
    test("chess.engine importable", True)

    # Verify piece images can be loaded by pygame
    pygame.init()
    piece_path = os.path.join("assets", "pieces", "wK.png")
    img_test = pygame.image.load(piece_path)
    test("Piece image loadable by pygame", img_test is not None)
    pygame.quit()
except Exception as e:
    test(f"Chess GUI imports ({e})", False)

# ---------- SUMMARY ----------
print(f"\n{'='*50}")
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED - see above for details")
    sys.exit(1)
