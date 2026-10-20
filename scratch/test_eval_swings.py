import os
import sys
import time
import pygame
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess_gui

def test_swings():
    print("Testing Eval Trajectory with eval swings (White & Black advantage)...")

    chess_gui.board.reset()
    with chess_gui.analysis_lock:
        chess_gui.eval_history.clear()

    chess_gui.trigger_analysis()
    time.sleep(0.5)

    # Sequence of moves featuring an eval swing
    moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "g6", "Qf3", "Nf6", "g4", "Nxg4", "Qxg4", "d5"]
    
    for san in moves:
        move_obj = chess_gui.board.parse_san(san)
        chess_gui.board.push(move_obj)
        chess_gui.trigger_analysis(san)
        time.sleep(0.5)

    time.sleep(1.0)

    with chess_gui.analysis_lock:
        hist = list(chess_gui.eval_history)
    print(f"Recorded history ({len(hist)} entries): {hist}")

    # Render frame
    chess_gui.draw_board()
    chess_gui.draw_pieces()
    chess_gui.draw_best_move_arrow()
    chess_gui.draw_ghost_piece()
    chess_gui.draw_eval_history()
    chess_gui.draw_eval_bar()
    chess_gui.draw_commentary_panel()
    pygame.display.flip()

    screenshot_path = os.path.join(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f519ac23-4f03-4a92-afa7-347c34ea9a3f", "eval_trajectory_swings.png")
    pygame.image.save(chess_gui.screen, screenshot_path)
    print(f"Saved screenshot to: {screenshot_path}")

    chess_gui.engine.quit()
    pygame.quit()

if __name__ == "__main__":
    test_swings()
