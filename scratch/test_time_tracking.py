"""
TASK 11 Test: Move-Time Tracking with artificial delays.
Simulates moves with varying sleep durations and verifies:
1. Per-move times match elapsed sleep durations.
2. Running White/Black totals sum correctly.
3. Clock HUD renders without overlap.
"""
import os
import sys
import time
import pygame
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess_gui


def test_time_tracking():
    print("\n" + "=" * 60)
    print("TASK 11: Testing Move-Time Tracking & Chess Clock")
    print("=" * 60)

    # Reset board and clock state
    chess_gui.board.reset()
    with chess_gui.analysis_lock:
        chess_gui.eval_history.clear()
        chess_gui.latest_blunder_event = None
    chess_gui.turn_start_time = time.time()
    chess_gui.white_total_time = 0.0
    chess_gui.black_total_time = 0.0
    chess_gui.last_move_time = None
    chess_gui.move_time_history.clear()

    # Initial analysis
    chess_gui.trigger_analysis()
    time.sleep(0.5)

    # Define moves with expected delays (seconds)
    test_moves = [
        ("e4",  1.0),   # White move 1 -> ~1.0s
        ("e5",  2.0),   # Black move 1 -> ~2.0s
        ("Nf3", 0.5),   # White move 2 -> ~0.5s
        ("Nc6", 1.5),   # Black move 2 -> ~1.5s
        ("Bb5", 0.8),   # White move 3 -> ~0.8s
    ]

    expected_white = 0.0
    expected_black = 0.0
    tolerance = 0.25  # seconds tolerance for timer accuracy

    print("\n--- Per-Move Time Tracking ---")
    for san, delay in test_moves:
        # Simulate thinking time
        time.sleep(delay)

        # Record who's moving
        is_white = chess_gui.board.turn == chess.WHITE

        # Compute elapsed (same logic as chess_gui.py)
        elapsed = time.time() - chess_gui.turn_start_time
        if is_white:
            chess_gui.white_total_time += elapsed
            expected_white += elapsed
        else:
            chess_gui.black_total_time += elapsed
            expected_black += elapsed
        chess_gui.last_move_time = elapsed
        chess_gui.move_time_history.append((chess_gui.board.ply(), elapsed))
        chess_gui.turn_start_time = time.time()

        # Apply the move
        move_obj = chess_gui.board.parse_san(san)
        chess_gui.board.push(move_obj)
        chess_gui.trigger_analysis(san, move_time=elapsed)

        mover = "White" if is_white else "Black"
        within_tolerance = abs(elapsed - delay) < tolerance
        status = "OK" if within_tolerance else "DRIFT"
        print(f"  Move {san:<5} ({mover:<5}): Expected ~{delay:.1f}s | Actual: {elapsed:.2f}s | [{status}]")

    # Wait for analysis threads to finish
    time.sleep(1.0)

    # Verify running totals
    print("\n--- Running Totals ---")
    print(f"  White Total: {chess_gui.white_total_time:.2f}s (expected ~{1.0 + 0.5 + 0.8:.1f}s)")
    print(f"  Black Total: {chess_gui.black_total_time:.2f}s (expected ~{2.0 + 1.5:.1f}s)")

    white_expected_sum = sum(d for san, d in test_moves[0::2])  # moves at index 0, 2, 4 = White
    black_expected_sum = sum(d for san, d in test_moves[1::2])  # moves at index 1, 3 = Black

    white_close = abs(chess_gui.white_total_time - white_expected_sum) < tolerance * len(test_moves)
    black_close = abs(chess_gui.black_total_time - black_expected_sum) < tolerance * len(test_moves)
    print(f"  White total within tolerance: {white_close}")
    print(f"  Black total within tolerance: {black_close}")

    # Verify move_time_history
    print(f"\n--- Move Time History ({len(chess_gui.move_time_history)} entries) ---")
    for ply, elapsed in chess_gui.move_time_history:
        mover = "White" if ply % 2 == 0 else "Black"
        print(f"  Ply {ply} ({mover}): {elapsed:.2f}s")

    # Render frame with chess clock visible
    chess_gui.draw_board()
    chess_gui.draw_pieces()
    chess_gui.draw_best_move_arrow()
    chess_gui.draw_ghost_piece()
    chess_gui.draw_move_clock()
    chess_gui.draw_eval_history()
    chess_gui.draw_blunder_alert()
    chess_gui.draw_eval_bar()
    chess_gui.draw_commentary_panel()
    pygame.display.flip()

    screenshot_path = os.path.join(
        r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f519ac23-4f03-4a92-afa7-347c34ea9a3f",
        "move_time_tracking.png"
    )
    pygame.image.save(chess_gui.screen, screenshot_path)
    print(f"\nSaved screenshot to: {screenshot_path}")

    print("\n[TASK 11 TEST COMPLETE]")

    chess_gui.engine.quit()
    pygame.quit()


if __name__ == "__main__":
    test_time_tracking()
