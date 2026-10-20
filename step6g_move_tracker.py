"""
AI Chess Vision Coach - Step 6g: Incremental Move Tracking via Legal Move Matching

Goal:
Instead of relying on fragile 2D visual piece-type classification, we track game progress
incrementally by detecting WHICH SQUARES CHANGED (occupancy + piece color) between two board states,
and finding the matching legal move in python-chess.
"""

import cv2
import numpy as np
import chess
from piece_color import warp_board


class BoardStateTracker:
    """
    Maintains the authoritative game state wrapping a python-chess Board object
    and tracks the 8x8 vision grid (is_occupied, piece_color).
    """

    def __init__(self, fen=None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()

    def get_vision_grid(self):
        """
        Generates the current 8x8 vision grid from the internal chess.Board state.
        Mapping: square_name (e.g. 'e4') -> (is_occupied: bool, color: 'white'/'black'/None)
        """
        return board_to_grid(self.board)

    def apply_move(self, move):
        """Pushes a move onto the internal authoritative chess board."""
        self.board.push(move)


def board_to_grid(board):
    """
    Converts a chess.Board state into an 8x8 vision grid.
    Returns dict mapping sq_name -> (is_occupied, color_str).
    """
    grid = {}
    piece_map = board.piece_map()
    for square_idx in range(64):
        sq_name = chess.square_name(square_idx)
        piece = piece_map.get(square_idx)
        if piece is None:
            grid[sq_name] = (False, None)
        else:
            color_str = "white" if piece.color == chess.WHITE else "black"
            grid[sq_name] = (True, color_str)
    return grid


def diff_grids(old_grid, new_grid):
    """
    PART B: Diff Detection
    Compares two vision grids (old_grid vs new_grid) and returns a sorted list
    of square names whose (is_occupied, piece_color) state changed.
    """
    changed_squares = []
    for sq_name in old_grid:
        if old_grid[sq_name] != new_grid.get(sq_name):
            changed_squares.append(sq_name)
    return sorted(changed_squares)


def infer_move(board, changed_squares):
    """
    PART C: Move Inference via Legal-Move Matching
    """
    old_grid = board_to_grid(board)
    target_set = set(changed_squares)

    matching_moves = []

    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        simulated_grid = board_to_grid(board_copy)

        sim_diff = set(diff_grids(old_grid, simulated_grid))

        if sim_diff == target_set:
            matching_moves.append(move)

    if len(matching_moves) == 0:
        return None, "NO_MATCH: No legal move matches the observed vision grid diff.", []

    elif len(matching_moves) == 1:
        return matching_moves[0], "SINGLE_MATCH", matching_moves

    else:
        queen_promos = [m for m in matching_moves if m.promotion == chess.QUEEN]
        selected_move = queen_promos[0] if queen_promos else matching_moves[0]

        candidates_san = [board.san(m) for m in matching_moves]
        msg = (f"AMBIGUOUS_MATCH: {len(matching_moves)} legal candidates match diff "
               f"({', '.join(candidates_san)}). Defaulting to Queen promotion ({board.san(selected_move)}).")
        return selected_move, msg, matching_moves


def capture_grid_from_photo(image_path, empty_image_path="chessboard.jpg", corners=None):
    """
    PART E: Hook point for Real Vision (occupancy + piece color grid extraction)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    if corners is None:
        empty_img = cv2.imread(empty_image_path)
        if empty_img is None:
            empty_img = cv2.imread("chessboard1.jpg")
        empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(empty_gray, (7, 7), None)
        if not found:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS)
            found, corners = cv2.findChessboardCorners(empty_gray, (7, 7), flags)
        if not found:
            found, corners = cv2.findChessboardCorners(empty_gray, (7, 7), None)
        if not found:
            raise RuntimeError("Could not find calibration corners.")

    warped = warp_board(image, corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sq_size = warped.shape[0] // 8

    vision_grid = {}

    for row in range(8):
        for col in range(8):
            sq_name = f"{'abcdefgh'[col]}{8 - row}"

            crop = warped_gray[row*sq_size:(row+1)*sq_size, col*sq_size:(col+1)*sq_size]
            var = np.var(crop)
            is_occupied = var > 200.0

            if not is_occupied:
                vision_grid[sq_name] = (False, None)
            else:
                cx, cy = sq_size // 2, sq_size // 2
                radius = int(sq_size * 0.15)
                mask = np.zeros_like(crop, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                center_brightness = cv2.mean(crop, mask=mask)[0]

                color_str = "white" if center_brightness > 120.0 else "black"
                vision_grid[sq_name] = (True, color_str)

    return vision_grid


def test_simulated_move_tracking():
    ref_board = chess.Board()
    tracker = BoardStateTracker()
    test_moves_san = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6", "O-O", "Nf6", "Re1", "Bg4", "h3"]

    correct_inferences = 0
    total_moves = len(test_moves_san)

    print(f"{'#':<3} | {'Move (SAN)':<10} | {'Observed Diff Squares':<25} | {'Inferred Move':<15} | {'Status'}")
    print("-" * 75)

    for idx, san_move in enumerate(test_moves_san, 1):
        old_grid = tracker.get_vision_grid()
        move_obj = ref_board.parse_san(san_move)
        ref_board.push(move_obj)
        new_grid = board_to_grid(ref_board)
        diff_sqs = diff_grids(old_grid, new_grid)

        inferred_move, status, candidates = infer_move(tracker.board, diff_sqs)

        if inferred_move == move_obj:
            correct_inferences += 1
            result_str = "MATCH"
        else:
            result_str = f"MISMATCH (Got {inferred_move})"

        diff_str = ", ".join(diff_sqs)
        inferred_san = tracker.board.san(inferred_move) if inferred_move else "None"
        print(f"{idx:<3} | {san_move:<10} | {diff_str:<25} | {inferred_san:<15} | {result_str}")
        tracker.apply_move(inferred_move)

    print(f"Tracking Accuracy: {correct_inferences}/{total_moves} ({(correct_inferences/total_moves)*100:.2f}%)")


if __name__ == "__main__":
    test_simulated_move_tracking()
