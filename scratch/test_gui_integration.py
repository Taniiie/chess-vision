import sys
sys.path.append('.')
import chess_gui

def test_gui_wiring():
    print("Testing GUI Wiring & Integration Test with Groq LLM Commentary...")
    chess_gui.run_simulated_integration_test()

if __name__ == "__main__":
    test_gui_wiring()
