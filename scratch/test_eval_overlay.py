import os
import sys
import time
import pygame
import chess

# Ensure local dir in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess_gui

def test_eval_overlay():
    print("Testing Eval Trajectory Overlay...")
    
    # 1. Trigger simulated integration test to populate history
    chess_gui.run_simulated_integration_test()
    
    # Wait briefly for background stockfish analysis threads to finish
    time.sleep(2.0)
    
    # Check eval_history content
    with chess_gui.analysis_lock:
        hist = list(chess_gui.eval_history)
    print(f"Recorded eval history entries ({len(hist)}): {hist}")

    # Render frame to pygame screen surface
    chess_gui.draw_board()
    chess_gui.draw_pieces()
    chess_gui.draw_best_move_arrow()
    chess_gui.draw_ghost_piece()
    chess_gui.draw_eval_history()
    chess_gui.draw_eval_bar()
    chess_gui.draw_commentary_panel()
    pygame.display.flip()

    # Save screenshot of full window
    screenshot_path = os.path.join(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f519ac23-4f03-4a92-afa7-347c34ea9a3f", "eval_trajectory_overlay.png")
    pygame.image.save(chess_gui.screen, screenshot_path)
    print(f"Saved screenshot to: {screenshot_path}")

    chess_gui.engine.quit()
    pygame.quit()

if __name__ == "__main__":
    test_eval_overlay()
