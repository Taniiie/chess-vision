"""
AI Chess Vision Coach - Step 7: LLM Commentary & Text-to-Speech Voice Engine

Features:
1. Groq API integration (llama-3.1-8b-instant) for real-time 1-2 sentence AI coach commentary.
2. Threaded offline text-to-speech using pyttsx3.
3. Robust error handling, API key safety (reading from .env), and response caching.
"""

import os
import queue
import threading
import time

from dotenv import load_dotenv
import chess
from groq import Groq
import pyttsx3

# Load environment variables from root .env file
load_dotenv()

# Response cache to prevent redundant API calls
_commentary_cache = {}


def normalize_text(text):
    """Normalizes unicode hyphens, quotes, and symbols to ASCII for clean console/TTS compatibility."""
    if not text:
        return text
    replacements = {
        "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "..."
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode('ascii', 'ignore').decode('ascii')


class VoiceCoachEngine:
    """
    PART B: Text-to-Speech Engine Manager
    Runs pyttsx3 in a dedicated background worker thread so speech synthesis
    never blocks or freezes the Pygame main loop.
    """

    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Background thread worker for TTS audio output."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)   # Speaking rate (WPM)
            engine.setProperty("volume", 1.0) # Volume (0.0 to 1.0)
        except Exception as e:
            print(f"[TTS Warning] Failed to initialize pyttsx3 engine: {e}")
            engine = None

        while not self._stop_event.is_set():
            try:
                text = self.speech_queue.get(timeout=0.2)
                if text and engine:
                    self.is_speaking = True
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        print(f"[TTS Speech Error] {err}")
                    finally:
                        self.is_speaking = False
                    self.speech_queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text):
        """Queues commentary text for non-blocking asynchronous speech."""
        if not text:
            return
        # Clear any old un-spoken queued speech if player moves quickly
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        self.speech_queue.put(text)


# Global singleton instance of VoiceCoachEngine
voice_coach = VoiceCoachEngine()


def generate_commentary(board, last_move_san, eval_score_str, best_move_san=None, blunder_context=None, move_time=None):
    """
    PART A: Commentary Generation via Groq API (llama-3.1-8b-instant)
    
    Parameters:
    - board: chess.Board object
    - last_move_san: str (e.g., "Nf3", "Bxc6", "O-O")
    - eval_score_str: str (e.g., "+0.3", "-1.5", "Mate in 2")
    - best_move_san: str (optional), Stockfish suggested move for next turn
    - blunder_context: str (optional), 'blunder' or 'brilliant'
    - move_time: float (optional), seconds taken on this move
    
    Returns:
    - commentary_text: str (1-2 sentence spoken coach commentary)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[LLM Notice] GROQ_API_KEY not found in environment. Using fallback commentary.")
        moved_color = "Black" if board.turn == chess.WHITE else "White"
        if blunder_context == "blunder":
            return f"Ouch! {moved_color} blundered with {last_move_san}."
        elif blunder_context == "brilliant":
            return f"Brilliant move by {moved_color} with {last_move_san}!"
        return f"{moved_color} played {last_move_san}."

    # Cache key check
    cache_key = f"{board.fen()}_{last_move_san}_{eval_score_str}_{best_move_san}_{blunder_context}"
    if cache_key in _commentary_cache:
        return _commentary_cache[cache_key]

    # Context indicators
    is_check = board.is_check()
    is_checkmate = board.is_checkmate()
    turn_color = "White" if board.turn == chess.WHITE else "Black"
    moved_color = "Black" if turn_color == "White" else "White"

    # Build prompt for LLM
    prompt = (
        f"{moved_color} just played '{last_move_san}'. "
        f"The current evaluation is {eval_score_str}. "
    )
    if blunder_context == "blunder":
        prompt += f"CRITICAL: {moved_color}'s move '{last_move_san}' was a HUGE BLUNDER causing a sharp evaluation drop! React with shock (e.g., 'Ouch, that's a blunder!') and briefly explain the mistake. "
    elif blunder_context == "brilliant":
        prompt += f"CRITICAL: {moved_color}'s move '{last_move_san}' was a BRILLIANT MOVE! Express excitement and praise the move! "

    # Task 11: Long think context for voice coach
    if move_time is not None and move_time >= 15.0:
        prompt += f"Note: {moved_color} spent a long think of {move_time:.1f} seconds deciding this move. Naturally mention this long deliberation if relevant. "

    if is_checkmate:
        prompt += "This move is CHECKMATE! The game is over. "
    elif is_check:
        prompt += "This move delivers CHECK! "

    if best_move_san and not is_checkmate:
        prompt += f"Stockfish suggests the best move for {turn_color} is '{best_move_san}'. "

    GROQ_MODELS = [
        "groq/compound-mini",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ]

    client = Groq(api_key=api_key)
    commentary = None

    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly, knowledgeable AI Chess Coach giving live audio commentary. "
                            "Provide 1-2 SHORT, energetic, natural sentences analyzing the move that was just played "
                            "or pointing out what to watch out for. Keep it concise for voice broadcast. "
                            "Do NOT use markdown, asterisks, or bullet points."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=70,
            )

            commentary = response.choices[0].message.content.strip()
            commentary = normalize_text(commentary.replace("*", "").replace("#", ""))
            if commentary:
                break
        except Exception:
            continue

    if commentary:
        _commentary_cache[cache_key] = commentary
        return commentary
    else:
        fallback = f"{moved_color} played {last_move_san}. Current eval: {eval_score_str}."
        return normalize_text(fallback)


def test_commentary_situations():
    """
    PART D: Test generate_commentary() with hand-crafted example situations.
    """
    print("=" * 60)
    print("PART D: Testing LLM Commentary Engine with Example Situations")
    print("=" * 60)

    test_cases = [
        {
            "name": "1. Normal Opening Move",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "last_move": "e4",
            "eval": "+0.2",
            "best_move": "e5"
        },
        {
            "name": "2. Capture Move",
            "fen": "r1bqkbnr/1ppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
            "last_move": "Bxc6",
            "eval": "+0.5",
            "best_move": "dxc6"
        },
        {
            "name": "3. Checking Move",
            "fen": "r1bqk2r/pppp1ppp/2n5/1B2p3/1b2P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 2 5",
            "last_move": "Bb4+",
            "eval": "0.0",
            "best_move": "c3"
        },
        {
            "name": "4. Blunder / Evaluation Swing",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "last_move": "f6?",
            "eval": "+2.8 (White advantage)",
            "best_move": "Nf6"
        }
    ]

    for tc in test_cases:
        b = chess.Board(tc["fen"])
        commentary = generate_commentary(b, tc["last_move"], tc["eval"], tc["best_move"])
        print(f"\n--- {tc['name']} ---")
        print(f"Move Played: {tc['last_move']} | Eval: {tc['eval']} | Best Next: {tc['best_move']}")
        print(f"Generated LLM Commentary: \"{commentary}\"")

    print("\n" + "=" * 60)
    print("Commentary generation test complete.")
    print("=" * 60)


if __name__ == "__main__":
    test_commentary_situations()
