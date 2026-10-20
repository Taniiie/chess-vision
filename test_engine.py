import chess
import chess.engine

STOCKFISH_PATH = r"stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

info = engine.analyse(board, chess.engine.Limit(time=0.5))
print("Evaluation:", info["score"])
print("Best move:", engine.play(board, chess.engine.Limit(time=0.5)).move)

engine.quit()