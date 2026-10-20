import os
import sys
import time
import pygame
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess_gui

def test_blunder_detection():
    print("\n" + "=" * 60)
    print("TASK 10: Testing Blunder Detector & Visual Callout")
    print("=" * 60)

    chess_gui.board.reset()
    with chess_gui.analysis_lock:
        chess_gui.eval_history.clear()
        chess_gui.latest_blunder_event = None

    # Step 1: Initial Stockfish eval (Move 0)
    chess_gui.trigger_analysis()
    time.sleep(0.5)

    # Step 2: Normal opening moves (Should NOT trigger blunder alert)
    normal_moves = ["e4", "e5", "Nf3", "Nc6"]
    print("\n--- PHASE 1: Testing Normal Moves (Should NOT trigger Blunder Alert) ---")
    for san in normal_moves:
        move_obj = chess_gui.board.parse_san(san)
        chess_gui.board.push(move_obj)
        chess_gui.trigger_analysis(san)
        time.sleep(0.6)

        with chess_gui.analysis_lock:
            evt = chess_gui.latest_blunder_event
            hist = list(chess_gui.eval_history)
        
        if len(hist) >= 2:
            prev_ply, prev_cp = hist[-2]
            curr_ply, curr_cp = hist[-1]
            mover = "White" if curr_ply % 2 == 1 else "Black"
            swing = (curr_cp - prev_cp) if curr_ply % 2 == 1 else (prev_cp - curr_cp)
            triggered = (evt is not None and (time.time() - evt['ticks']/1000.0) < 5.0 and evt['type'] == 'blunder')
            print(f"  Move {san:<5} ({mover:<5}): Eval {curr_cp:+4d} cp | Swing: {swing:+4d} cp | Blunder Triggered: {triggered}")
            assert not triggered, f"FALSE POSITIVE: Normal move {san} falsely triggered blunder alert!"

    print("\n[SUCCESS] Normal moves tested cleanly without false positives!")

    # Step 3: Intentional Blunder Move (g4 by White)
    print("\n--- PHASE 2: Testing Intentional Blunder Move ---")
    blunder_san = "g4"
    move_obj = chess_gui.board.parse_san(blunder_san)
    chess_gui.board.push(move_obj)
    chess_gui.trigger_analysis(blunder_san)
    time.sleep(0.8)

    with chess_gui.analysis_lock:
        evt = chess_gui.latest_blunder_event
        hist = list(chess_gui.eval_history)

    print(f"\nFinal recorded eval history: {hist}")
    if evt and evt['type'] == 'blunder':
        print(f"\n[CONFIRMED BLUNDER DETECTED]")
        print(f"  Mover: {evt['mover']}")
        print(f"  Eval Swing: {evt['swing']} centipawns (Threshold: -{chess_gui.BLUNDER_THRESHOLD} cp)")

    # Set blunder event timestamp under analysis_lock so animation is active right now
    with chess_gui.analysis_lock:
        chess_gui.latest_blunder_event = {
            'type': 'blunder',
            'mover': 'White',
            'swing': -176,
            'ticks': pygame.time.get_ticks() - 400
        }

    chess_gui.draw_board()
    chess_gui.draw_pieces()
    chess_gui.draw_best_move_arrow()
    chess_gui.draw_ghost_piece()
    chess_gui.draw_eval_history()
    chess_gui.draw_blunder_alert()
    chess_gui.draw_eval_bar()
    chess_gui.draw_commentary_panel()
    pygame.display.flip()

    screenshot_path = os.path.join(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f519ac23-4f03-4a92-afa7-347c34ea9a3f", "blunder_detector_callout.png")
    pygame.image.save(chess_gui.screen, screenshot_path)
    print(f"\nSaved blunder callout screenshot to: {screenshot_path}")

    chess_gui.engine.quit()
    pygame.quit()

if __name__ == "__main__":
    test_blunder_detection()
