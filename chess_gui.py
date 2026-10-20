"""
AI Chess Vision Coach - Complete System (Steps 1-7 + Step 6h + Task 8 AR Ghost Piece)

Features:
1. Mouse-driven play (interactive GUI).
2. Photo Move Mode:
   - 'i' key: Calibrate using an EMPTY board photo.
   - 'm' key: Input a NEW board photo path to detect and apply moves.
   - 't' key: Run simulated end-to-end integration test.
   - 'r' key: Reset board position.
   - 'g' key: Toggle AR Ghost Piece floating overlay on/off.
3. AR Overlay: Live Stockfish evaluation bar + engine best-move arrow + floating 3D-illusion Ghost Piece.
4. LLM Voice Coach:
   - Real-time commentary generated via Groq API (llama-3.1-8b-instant / fallback models).
   - Non-blocking offline text-to-speech audio via pyttsx3.
   - Live on-screen AI Coach commentary box + speaking status indicator ("🔊 Speaking..." vs "🎙️ Coach Idle").
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import time
import threading
import cv2
import numpy as np
import pygame
import chess
import chess.engine

from step6g_move_tracker import (
    BoardStateTracker,
    board_to_grid,
    diff_grids,
    infer_move,
    capture_grid_from_photo
)
from step7_commentary import (
    generate_commentary,
    voice_coach,
    normalize_text
)

# ---------- CONFIG ----------
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
EVAL_BAR_WIDTH = 40
COMMENTARY_PANEL_HEIGHT = 75
WIDTH = BOARD_SIZE + EVAL_BAR_WIDTH
HEIGHT = BOARD_SIZE + COMMENTARY_PANEL_HEIGHT

PIECES_DIR = "assets/pieces"
STOCKFISH_PATH = r"stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
ANALYSIS_TIME = 0.3  # seconds per evaluation

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (246, 246, 105)
ARROW_COLOR = (0, 150, 60)
EVAL_BAR_WHITE = (240, 240, 240)
EVAL_BAR_BLACK = (30, 30, 30)
EVAL_CAP = 600

# TASK 10: Blunder & Brilliant Move Thresholds (Centipawns)
BLUNDER_THRESHOLD = 150   # 1.5 pawns drop for the mover
BRILLIANT_THRESHOLD = 150 # 1.5 pawns gain for the mover

# TASK 11: Move-Time Tracking Config
LONG_THINK_THRESHOLD = 15.0  # seconds - moves taking longer get noted by voice coach

# ---------- INIT ----------
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Chess Vision Coach - AR GUI & Floating Ghost Piece")
clock = pygame.time.Clock()

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Fonts
font_title = pygame.font.SysFont("arial", 13, bold=True)
font_body = pygame.font.SysFont("arial", 14)
font_status = pygame.font.SysFont("arial", 12, bold=True)

# ---------- PHOTO MODE & AR GLOBAL STATE ----------
photo_tracker = None      # BoardStateTracker instance
saved_corners = None      # Board calibration corners
last_known_grid = None    # Last known 8x8 vision grid
show_ghost_piece = True   # Toggleable AR Ghost Piece overlay
show_eval_history = True  # Toggleable Eval Trajectory Overlay graph ('h' key)
show_blunder_alerts = True # Toggleable Blunder Detector overlay ('b' key)
show_move_clock = True     # Toggleable Chess Clock overlay ('c' key)

# ---------- SHARED ANALYSIS & COMMENTARY STATE ----------
analysis_lock = threading.Lock()
latest_score = None            # chess.engine.Score, from White's perspective
latest_best_move = None        # chess.Move
latest_commentary_text = "Welcome to AI Chess Vision Coach! Make a move or press 'm' for Photo Move."
eval_history = []              # Running history: [(ply, score_cp), ...]
latest_blunder_event = None    # Dict storing active blunder animation state

# ---------- TASK 11: MOVE-TIME TRACKING STATE ----------
turn_start_time = time.time()  # Timestamp when current player's turn began
white_total_time = 0.0         # Running total thinking time for White
black_total_time = 0.0         # Running total thinking time for Black
last_move_time = None          # Seconds taken on the most recent move
move_time_history = []         # Per-move time history: [(ply, elapsed_secs), ...]


def format_score(score):
    """Formats chess.engine.Score into human-readable eval string."""
    if score is None:
        return "0.0"
    if score.is_mate():
        m = score.mate()
        return f"Mate in {abs(m)}"
    cp = score.score()
    pawns = cp / 100.0
    if pawns > 0:
        return f"+{pawns:.1f}"
    elif pawns < 0:
        return f"{pawns:.1f}"
    else:
        return "0.0"


def background_analyze(position_fen, last_move_san=None, move_time=None):
    """Runs Stockfish evaluation + LLM Commentary in background thread."""
    global latest_score, latest_best_move, latest_commentary_text

    temp_board = chess.Board(position_fen)
    if temp_board.is_game_over():
        return

    with analysis_lock:
        info = engine.analyse(temp_board, chess.engine.Limit(time=ANALYSIS_TIME))
        score = info["score"].white()
        best_move = engine.play(temp_board, chess.engine.Limit(time=ANALYSIS_TIME)).move

        latest_score = score
        latest_best_move = best_move

        # Part A: Append eval score to running eval_history list
        if score is not None:
            if score.is_mate():
                m = score.mate()
                cp_val = 1000 if (m is not None and m > 0) else -1000
            else:
                cp_val = score.score()
                if cp_val is None:
                    cp_val = 0
        else:
            cp_val = 0

        eval_history.append((temp_board.ply(), cp_val))

        # Part A (Task 10): Blunder & Brilliant Move Detection Logic
        blunder_type = None
        if len(eval_history) >= 2:
            prev_ply, prev_cp = eval_history[-2]
            curr_ply, curr_cp = eval_history[-1]

            # Mover is White if curr_ply is odd (1, 3, 5...), Black if even (2, 4, 6...)
            mover = "White" if curr_ply % 2 == 1 else "Black"
            mover_swing = (curr_cp - prev_cp) if curr_ply % 2 == 1 else (prev_cp - curr_cp)

            if mover_swing <= -BLUNDER_THRESHOLD:
                blunder_type = "blunder"
                latest_blunder_event = {
                    'type': 'blunder',
                    'mover': mover,
                    'swing': mover_swing,
                    'ticks': pygame.time.get_ticks()
                }
                print(f"\n  [BLUNDER DETECTED!] {mover} blundered with move! Swing: {mover_swing} centipawns.")
            elif mover_swing >= BRILLIANT_THRESHOLD:
                blunder_type = "brilliant"
                latest_blunder_event = {
                    'type': 'brilliant',
                    'mover': mover,
                    'swing': mover_swing,
                    'ticks': pygame.time.get_ticks()
                }
                print(f"\n  [BRILLIANT MOVE DETECTED!] {mover} played a brilliant move! Swing: +{mover_swing} centipawns.")

        eval_str = format_score(score)
        best_move_san = temp_board.san(best_move) if (best_move and best_move in temp_board.legal_moves) else None

    print(f"  [Stockfish] Eval: {score}  |  Best Move Suggestion: {best_move_san}")

    # Generate LLM Commentary & Speak Aloud if a move was played
    if last_move_san:
        commentary = generate_commentary(temp_board, last_move_san, eval_str, best_move_san, blunder_context=blunder_type, move_time=move_time)
        with analysis_lock:
            latest_commentary_text = commentary
        print(f"  [AI Coach Spoke]: \"{commentary}\"")
        voice_coach.speak(commentary)


def trigger_analysis(last_move_san=None, move_time=None):
    """Kicks off Stockfish analysis + LLM commentary for current position."""
    fen = board.fen()
    thread = threading.Thread(target=background_analyze, args=(fen, last_move_san, move_time), daemon=True)
    thread.start()


def load_piece_images():
    """Loads piece PNGs from PIECES_DIR."""
    images = {}
    piece_codes = ["P", "N", "B", "R", "Q", "K"]
    for color in ["w", "b"]:
        for code in piece_codes:
            filename = f"{color}{code}.png"
            path = os.path.join(PIECES_DIR, filename)
            if os.path.exists(path):
                img = pygame.image.load(path)
                img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                images[color + code] = img
            else:
                print(f"[warning] missing piece image: {path}")
    return images


PIECE_IMAGES = load_piece_images()


def square_to_pixel(square):
    col = chess.square_file(square)
    row = 7 - chess.square_rank(square)
    return col * SQUARE_SIZE, row * SQUARE_SIZE


def square_center(square):
    x, y = square_to_pixel(square)
    return x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2


def pixel_to_square(pos):
    x, y = pos
    if x >= BOARD_SIZE:
        return None
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    if row < 0 or row > 7 or col < 0 or col > 7:
        return None
    return chess.square(col, row)


def draw_board(selected_square=None):
    for rank in range(8):
        for file in range(8):
            color = LIGHT_COLOR if (rank + file) % 2 == 0 else DARK_COLOR
            x = file * SQUARE_SIZE
            y = rank * SQUARE_SIZE
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

    if selected_square is not None:
        x, y = square_to_pixel(selected_square)
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, (x, y, SQUARE_SIZE, SQUARE_SIZE), 4)


def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        color_prefix = "w" if piece.color == chess.WHITE else "b"
        key = color_prefix + piece.symbol().upper()
        img = PIECE_IMAGES.get(key)
        if img:
            x, y = square_to_pixel(square)
            screen.blit(img, (x, y))


def draw_arrow(start_pos, end_pos, color, width=6):
    pygame.draw.line(screen, color, start_pos, end_pos, width)

    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    arrow_len = 18
    arrow_angle = math.radians(25)

    p1 = (
        end_pos[0] - arrow_len * math.cos(angle - arrow_angle),
        end_pos[1] - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        end_pos[0] - arrow_len * math.cos(angle + arrow_angle),
        end_pos[1] - arrow_len * math.sin(angle + arrow_angle),
    )
    pygame.draw.polygon(screen, color, [end_pos, p1, p2])


def draw_best_move_arrow():
    with analysis_lock:
        best_move = latest_best_move
    if best_move is None:
        return
    if best_move not in board.legal_moves:
        return
    start = square_center(best_move.from_square)
    end = square_center(best_move.to_square)
    draw_arrow(start, end, ARROW_COLOR)


def draw_ghost_piece():
    """
    TASK 8: Floating "Ghost Piece" AR Overlay for Best-Move Suggestion
    
    Draws a 2D floating illusion of the suggested piece on its destination square:
    1. Soft Shadow: A dark, semi-transparent ellipse centered on the destination square.
    2. Translucent Ghost Piece: Scaled to 125% size with 45% opacity.
    3. Gentle Bobbing & Opacity Pulse Animation: Sine-wave vertical offset + pulsing alpha.
    """
    if not show_ghost_piece:
        return

    with analysis_lock:
        best_move = latest_best_move

    if best_move is None or best_move not in board.legal_moves:
        return

    from_sq = best_move.from_square
    to_sq = best_move.to_square

    moved_piece = board.piece_at(from_sq)
    if moved_piece is None:
        return

    color_prefix = "w" if moved_piece.color == chess.WHITE else "b"
    key = color_prefix + moved_piece.symbol().upper()
    base_img = PIECE_IMAGES.get(key)
    if base_img is None:
        return

    dest_cx, dest_cy = square_center(to_sq)

    # 1. Soft Shadow sitting on the board
    shadow_w, shadow_h = int(SQUARE_SIZE * 0.65), int(SQUARE_SIZE * 0.28)
    shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (20, 20, 25, 110), (0, 0, shadow_w, shadow_h))

    shadow_x = dest_cx - shadow_w // 2
    shadow_y = dest_cy + int(SQUARE_SIZE * 0.12)
    screen.blit(shadow_surf, (shadow_x, shadow_y))

    # 2. Gentle bobbing animation & pulsing alpha
    ticks = pygame.time.get_ticks()
    bob_offset = math.sin(ticks * 0.005) * 5.0
    float_y = 18 + bob_offset  # 18px base upward float offset

    alpha_pulse = 120 + int(math.sin(ticks * 0.004) * 20)  # Opacity pulses between 100 and 140

    # 3. Floating Translucent Ghost Piece Image (125% scale)
    ghost_w = int(SQUARE_SIZE * 1.25)
    ghost_h = int(SQUARE_SIZE * 1.25)
    scaled_ghost = pygame.transform.smoothscale(base_img, (ghost_w, ghost_h))
    ghost_surf = scaled_ghost.copy()
    ghost_surf.set_alpha(alpha_pulse)

    ghost_x = dest_cx - ghost_w // 2
    ghost_y = dest_cy - ghost_h // 2 - int(float_y)

    screen.blit(ghost_surf, (ghost_x, ghost_y))


def draw_eval_history():
    """
    TASK 9: Eval Trajectory Overlay (Mini Momentum Chart)
    
    Plots eval_history move-by-move on a small semi-transparent HUD card overlay.
    - Horizontal line at eval=0
    - Clamped Y-axis [-500, +500] centipawns
    - Green-ish fill & line above zero line (White advantage)
    - Red-ish fill & line below zero line (Black advantage)
    - Current score badge & move count labels
    """
    if not show_eval_history:
        return

    with analysis_lock:
        history = list(eval_history)

    box_w, box_h = 210, 110
    box_x = BOARD_SIZE - box_w - 10  # 420px (10px gap before eval bar at 640px)
    box_y = 10                       # Top-right corner of the board

    card_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)

    # 1. Semi-transparent dark background card with subtle border
    pygame.draw.rect(card_surf, (20, 24, 33, 215), (0, 0, box_w, box_h), border_radius=8)
    pygame.draw.rect(card_surf, (60, 70, 90, 230), (0, 0, box_w, box_h), width=1, border_radius=8)

    # 2. Header title & toggle key badge
    font_graph_title = pygame.font.SysFont("arial", 11, bold=True)
    font_graph_sub = pygame.font.SysFont("arial", 10)

    title_txt = font_graph_title.render("EVAL TRAJECTORY", True, (200, 210, 225))
    key_txt = font_graph_sub.render("[H]", True, (120, 130, 150))
    card_surf.blit(title_txt, (10, 6))
    card_surf.blit(key_txt, (10 + title_txt.get_width() + 6, 7))

    # Current eval badge at top-right of box
    if history:
        last_cp = history[-1][1]
        if abs(last_cp) >= 1000:
            score_lbl = "MATE"
            badge_color = (0, 180, 90) if last_cp > 0 else (220, 50, 50)
        else:
            pawns = last_cp / 100.0
            score_lbl = f"+{pawns:.1f}" if pawns > 0 else f"{pawns:.1f}"
            if pawns > 0:
                badge_color = (0, 160, 80)
            elif pawns < 0:
                badge_color = (190, 50, 50)
            else:
                badge_color = (80, 90, 105)

        badge_surf = font_graph_title.render(score_lbl, True, (255, 255, 255))
        badge_bg_w = badge_surf.get_width() + 8
        badge_rect = pygame.Rect(box_w - badge_bg_w - 8, 4, badge_bg_w, 16)
        pygame.draw.rect(card_surf, badge_color, badge_rect, border_radius=3)
        card_surf.blit(badge_surf, (badge_rect.x + 4, badge_rect.y + 1))

    # Graph plot bounds inside card
    pad_left, pad_right = 26, 12
    pad_top, pad_bottom = 26, 18
    gw = box_w - pad_left - pad_right
    gh = box_h - pad_top - pad_bottom
    zero_y = pad_top + gh // 2

    # Draw eval=0 center line
    pygame.draw.line(card_surf, (80, 90, 105, 180), (pad_left, zero_y), (pad_left + gw, zero_y), 1)

    # Zero line label
    zero_lbl = font_graph_sub.render("0", True, (110, 120, 135))
    card_surf.blit(zero_lbl, (pad_left - 12, zero_y - 6))

    if not history:
        empty_txt = font_graph_sub.render("Analyzing...", True, (130, 140, 155))
        card_surf.blit(empty_txt, (pad_left + 40, zero_y - 6))
    elif len(history) == 1:
        # Single point dot
        cp_val = history[0][1]
        clamped = max(-EVAL_CAP, min(EVAL_CAP, cp_val))
        pt_y = zero_y - int((clamped / float(EVAL_CAP)) * (gh / 2.0))
        pt = (pad_left, pt_y)
        dot_color = (46, 204, 113) if cp_val >= 0 else (231, 76, 60)
        pygame.draw.circle(card_surf, dot_color, pt, 4)
    else:
        # Compute coordinates for all history points
        pts = []
        n = len(history)
        for idx, (ply, cp_val) in enumerate(history):
            x = pad_left + int(idx / float(n - 1) * gw)
            clamped = max(-EVAL_CAP, min(EVAL_CAP, cp_val))
            y = zero_y - int((clamped / float(EVAL_CAP)) * (gh / 2.0))
            pts.append((x, y))

        # Render filled polygons and colored line segments
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]

            # Intersect with zero_y if segment crosses center line
            if (p1[1] <= zero_y and p2[1] <= zero_y) or (p1[1] >= zero_y and p2[1] >= zero_y):
                segments = [(p1, p2)]
            else:
                # Calculate intersection x at zero_y
                dy = p2[1] - p1[1]
                t = (zero_y - p1[1]) / float(dy) if dy != 0 else 0
                mid_x = p1[0] + t * (p2[0] - p1[0])
                mid_pt = (mid_x, zero_y)
                segments = [(p1, mid_pt), (mid_pt, p2)]

            for sp1, sp2 in segments:
                avg_y = (sp1[1] + sp2[1]) / 2.0
                if avg_y <= zero_y:
                    # White advantage: Green
                    fill_poly = [(sp1[0], zero_y), sp1, sp2, (sp2[0], zero_y)]
                    pygame.draw.polygon(card_surf, (0, 200, 100, 45), fill_poly)
                    pygame.draw.line(card_surf, (46, 204, 113), sp1, sp2, 2)
                else:
                    # Black advantage: Red
                    fill_poly = [(sp1[0], zero_y), sp1, sp2, (sp2[0], zero_y)]
                    pygame.draw.polygon(card_surf, (220, 60, 60, 45), fill_poly)
                    pygame.draw.line(card_surf, (231, 76, 60), sp1, sp2, 2)

        # Small point markers
        for pt in pts:
            pygame.draw.circle(card_surf, (240, 245, 255), pt, 2)

        # X-axis move labels
        start_m = (history[0][0] + 1) // 2
        end_m = (history[-1][0] + 1) // 2
        m_start_lbl = font_graph_sub.render(f"M{start_m}", True, (130, 140, 155))
        m_end_lbl = font_graph_sub.render(f"M{end_m}", True, (130, 140, 155))
        card_surf.blit(m_start_lbl, (pad_left, box_h - 15))
        card_surf.blit(m_end_lbl, (pad_left + gw - m_end_lbl.get_width(), box_h - 15))

    screen.blit(card_surf, (box_x, box_y))


def draw_blunder_alert():
    """
    TASK 10: Dramatic Visual Moment for Blunders & Brilliant Moves
    
    1. Semi-transparent red flash/pulse over board area (1.5s fade in/out alpha animation).
    2. Bold high-contrast banner text: "💥 BLUNDER! (-X.X pawns) 💥" or "✨ BRILLIANT MOVE! ✨"
    3. Fully non-blocking (uses pygame tick delta).
    """
    if not show_blunder_alerts:
        return

    with analysis_lock:
        evt = latest_blunder_event

    if evt is None:
        return

    ticks = pygame.time.get_ticks()
    elapsed = ticks - evt['ticks']
    duration = 1500  # ms

    if elapsed > duration:
        return

    # Sine-wave alpha pulse: starts at 0, peaks at 130, fades to 0
    progress = elapsed / float(duration)
    alpha = int(130 * math.sin(math.pi * progress))

    is_blunder = (evt['type'] == 'blunder')

    # 1. Semi-transparent pulse overlay over board area (640x640)
    flash_surf = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    flash_color = (231, 76, 60, alpha) if is_blunder else (46, 204, 113, alpha)
    flash_surf.fill(flash_color)
    screen.blit(flash_surf, (0, 0))

    # 2. Bold central callout banner
    banner_w, banner_h = 370, 65
    banner_x = (BOARD_SIZE - banner_w) // 2
    banner_y = (BOARD_SIZE - banner_h) // 2

    banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)

    bg_color = (180, 30, 30, int(230 * math.sin(math.pi * progress))) if is_blunder else (20, 140, 70, int(230 * math.sin(math.pi * progress)))
    border_color = (255, 100, 100, alpha) if is_blunder else (100, 255, 180, alpha)

    pygame.draw.rect(banner_surf, bg_color, (0, 0, banner_w, banner_h), border_radius=10)
    pygame.draw.rect(banner_surf, border_color, (0, 0, banner_w, banner_h), width=3, border_radius=10)

    font_banner = pygame.font.SysFont("arial", 20, bold=True)
    font_sub = pygame.font.SysFont("arial", 12, bold=True)

    swing_pawns = abs(evt['swing']) / 100.0
    if is_blunder:
        title_text = f"BLUNDER! (-{swing_pawns:.1f} PAWNS)"
        sub_text = f"{evt['mover'].upper()} MADE A HUGE MISTAKE!"
    else:
        title_text = f"BRILLIANT MOVE! (+{swing_pawns:.1f} PAWNS)"
        sub_text = f"GREAT TACTICAL STRIKE BY {evt['mover'].upper()}!"

    txt_title = font_banner.render(title_text, True, (255, 255, 255))
    txt_sub = font_sub.render(sub_text, True, (255, 235, 180) if is_blunder else (220, 255, 230))

    banner_surf.blit(txt_title, ((banner_w - txt_title.get_width()) // 2, 10))
    banner_surf.blit(txt_sub, ((banner_w - txt_sub.get_width()) // 2, 38))

    screen.blit(banner_surf, (banner_x, banner_y))


def score_to_fraction(score):
    if score.is_mate():
        mate_in = score.mate()
        return 1.0 if mate_in > 0 else 0.0

    cp = score.score()
    cp = max(-EVAL_CAP, min(EVAL_CAP, cp))
    return (cp + EVAL_CAP) / (2 * EVAL_CAP)


def draw_eval_bar():
    bar_x = BOARD_SIZE
    with analysis_lock:
        score = latest_score

    fraction_white = score_to_fraction(score) if score is not None else 0.5

    white_height = int(BOARD_SIZE * fraction_white)
    black_height = BOARD_SIZE - white_height

    pygame.draw.rect(screen, EVAL_BAR_BLACK, (bar_x, 0, EVAL_BAR_WIDTH, black_height))
    pygame.draw.rect(screen, EVAL_BAR_WHITE, (bar_x, black_height, EVAL_BAR_WIDTH, white_height))


def draw_move_clock():
    """
    TASK 11: Chess Clock HUD Overlay
    
    Displays per-move and running total time for White and Black.
    Positioned at top-left of board (x=10, y=10) to avoid overlap with
    Eval Trajectory (top-right) and Eval Bar (right edge).
    """
    if not show_move_clock:
        return

    box_w, box_h = 185, 52
    box_x, box_y = 10, 10

    card_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)

    # Semi-transparent dark background
    pygame.draw.rect(card_surf, (20, 24, 33, 215), (0, 0, box_w, box_h), border_radius=8)
    pygame.draw.rect(card_surf, (60, 70, 90, 230), (0, 0, box_w, box_h), width=1, border_radius=8)

    font_clock_title = pygame.font.SysFont("arial", 11, bold=True)
    font_clock_val = pygame.font.SysFont("arial", 11)

    # Title
    title_txt = font_clock_title.render("CHESS CLOCK", True, (200, 210, 225))
    key_txt = font_clock_val.render("[C]", True, (120, 130, 150))
    card_surf.blit(title_txt, (8, 5))
    card_surf.blit(key_txt, (8 + title_txt.get_width() + 4, 5))

    # Last move time badge (top-right corner of card)
    if last_move_time is not None:
        last_lbl = f"Last: {last_move_time:.1f}s"
        last_color = (255, 200, 80) if last_move_time >= LONG_THINK_THRESHOLD else (180, 190, 200)
    else:
        last_lbl = "Last: --"
        last_color = (120, 130, 150)
    last_surf = font_clock_val.render(last_lbl, True, last_color)
    card_surf.blit(last_surf, (box_w - last_surf.get_width() - 8, 5))

    # Format time helper
    def fmt_time(secs):
        m = int(secs) // 60
        s = int(secs) % 60
        return f"{m:02d}:{s:02d}"

    # White clock
    w_lbl = font_clock_val.render(f"W: {fmt_time(white_total_time)}", True, (240, 240, 240))
    card_surf.blit(w_lbl, (10, 24))

    # Black clock
    b_lbl = font_clock_val.render(f"B: {fmt_time(black_total_time)}", True, (180, 180, 180))
    card_surf.blit(b_lbl, (10, 36))

    # Live elapsed for current turn (ticking clock)
    live_elapsed = time.time() - turn_start_time
    whose_turn = "White" if board.turn == chess.WHITE else "Black"
    live_txt = font_clock_val.render(f"{whose_turn}: {live_elapsed:.0f}s", True, (100, 220, 130))
    card_surf.blit(live_txt, (box_w - live_txt.get_width() - 8, 30))

    screen.blit(card_surf, (box_x, box_y))


def draw_commentary_panel():
    """Draws bottom AI Coach Commentary box + Voice Speaking Indicator."""
    panel_rect = pygame.Rect(0, BOARD_SIZE, WIDTH, COMMENTARY_PANEL_HEIGHT)
    pygame.draw.rect(screen, (25, 28, 36), panel_rect)
    pygame.draw.line(screen, (60, 65, 80), (0, BOARD_SIZE), (WIDTH, BOARD_SIZE), 2)

    with analysis_lock:
        text = latest_commentary_text

    # Header label
    lbl_title = font_title.render("AI CHESS COACH COMMENTARY:", True, (240, 200, 80))
    screen.blit(lbl_title, (12, BOARD_SIZE + 6))

    # Speaking status indicator badge
    is_speaking = voice_coach.is_speaking
    badge_bg = (0, 140, 60) if is_speaking else (60, 65, 80)
    badge_text = "Speaking..." if is_speaking else "Coach Idle"
    badge_surface = font_status.render(badge_text, True, (255, 255, 255))
    
    badge_rect = pygame.Rect(WIDTH - 110, BOARD_SIZE + 6, 98, 20)
    pygame.draw.rect(screen, badge_bg, badge_rect, border_radius=4)
    screen.blit(badge_surface, (badge_rect.x + (98 - badge_surface.get_width()) // 2, badge_rect.y + 2))

    # Wrap & render commentary text across 2 lines
    words = text.split(" ")
    line1, line2 = "", ""
    max_len = 75
    for w in words:
        if len(line1) + len(w) + 1 <= max_len:
            line1 += (w + " ")
        else:
            line2 += (w + " ")

    txt1 = font_body.render(line1.strip(), True, (220, 225, 235))
    screen.blit(txt1, (12, BOARD_SIZE + 26))

    if line2.strip():
        txt2 = font_body.render(line2.strip(), True, (220, 225, 235))
        screen.blit(txt2, (12, BOARD_SIZE + 48))


# ---------- PHOTO MODE CONTROLLER FUNCTIONS ----------

def initialize_photo_mode():
    global photo_tracker, saved_corners, last_known_grid
    print("\n" + "=" * 60)
    print("PHOTO MOVE MODE: CALIBRATION (PART A)")
    print("=" * 60)

    try:
        empty_path = input("Enter path to EMPTY board photo (default: 'chessboard.jpg'): ").strip()
    except EOFError:
        empty_path = "chessboard.jpg"

    if not empty_path:
        empty_path = "chessboard.jpg"

    if not os.path.exists(empty_path):
        print(f"[ERROR] Calibration failed: File not found at '{empty_path}'.")
        return False

    empty_img = cv2.imread(empty_path)
    if empty_img is None:
        print(f"[ERROR] Calibration failed: Could not decode image at '{empty_path}'.")
        return False

    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    
    found, corners = cv2.findChessboardCornersSB(empty_gray, (7, 7), None)
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(empty_gray, (7, 7), flags)
    if not found:
        found, corners = cv2.findChessboardCorners(empty_gray, (7, 7), None)

    if not found:
        print("[ERROR] Calibration failed: Could not detect chessboard corners.")
        return False

    saved_corners = corners
    photo_tracker = BoardStateTracker(board.fen())
    last_known_grid = photo_tracker.get_vision_grid()

    print("[SUCCESS] Board calibration saved! Photo Move Mode initialized.")
    print("  -> Press 'm' to submit a new board photo after making a move.\n")
    return True


def process_photo_move(image_path=None):
    global photo_tracker, saved_corners, last_known_grid

    if saved_corners is None or photo_tracker is None:
        print("[NOTICE] Photo Move Mode not calibrated yet. Initializing calibration...")
        success = initialize_photo_mode()
        if not success:
            return False

    if image_path is None:
        print("\n" + "=" * 60)
        print("MAKE MOVE VIA PHOTO (PART B)")
        print("=" * 60)
        try:
            image_path = input("Enter path to NEW board photo after making your move: ").strip()
        except EOFError:
            print("[ERROR] Input interrupted.")
            return False

    if not image_path or not os.path.exists(image_path):
        print(f"[ERROR] Photo file not found: '{image_path}'. Move cancelled.")
        return False

    try:
        new_grid = capture_grid_from_photo(image_path, corners=saved_corners)
    except Exception as e:
        print(f"[ERROR] Failed to extract vision grid from photo: {e}")
        return False

    diff_sqs = diff_grids(last_known_grid, new_grid)
    if not diff_sqs:
        print("[NOTICE] No square changes detected between last state and photo.")
        return False

    print(f"Detected vision grid changes on squares: {', '.join(diff_sqs)}")
    inferred_move, status, candidates = infer_move(board, diff_sqs)

    if inferred_move is None:
        print(f"\n[NO MATCH ERROR] {status}")
        print("  -> Game state was NOT modified. You may retry with a clearer photo.")
        return False

    # Handle Promotion Ambiguity
    if "AMBIGUOUS" in status and len(candidates) > 1:
        print(f"\n[AMBIGUITY DETECTED] Multiple legal promotions match this grid change:")
        for idx, cand in enumerate(candidates, 1):
            print(f"  {idx}. {board.san(cand)}")

        try:
            choice = input("Which piece did you promote to? (Q/R/B/N) [Default Q]: ").strip().upper()
        except EOFError:
            choice = "Q"

        promo_map = {'Q': chess.QUEEN, 'R': chess.ROOK, 'B': chess.BISHOP, 'N': chess.KNIGHT}
        selected_promo = promo_map.get(choice, chess.QUEEN)

        matching_promos = [m for m in candidates if m.promotion == selected_promo]
        if matching_promos:
            inferred_move = matching_promos[0]

    # UNIQUE MATCH or RESOLVED AMBIGUITY: Apply move
    global turn_start_time, white_total_time, black_total_time, last_move_time
    san_move = board.san(inferred_move)

    # Task 11: Record move time before applying
    elapsed = time.time() - turn_start_time
    is_white_move = board.turn == chess.WHITE
    if is_white_move:
        white_total_time += elapsed
    else:
        black_total_time += elapsed
    last_move_time = elapsed
    move_time_history.append((board.ply(), elapsed))
    turn_start_time = time.time()

    board.push(inferred_move)
    photo_tracker.apply_move(inferred_move)
    last_known_grid = new_grid

    print(f"\n[SUCCESS] Applied Photo Move: {san_move}  |  FEN: {board.fen()}")
    trigger_analysis(san_move, move_time=elapsed)
    return True


def run_simulated_integration_test():
    global photo_tracker, last_known_grid, turn_start_time, white_total_time, black_total_time, last_move_time
    print("\n" + "=" * 60)
    print("PART C: Running Simulated Integration Test on Real GUI Objects")
    print("=" * 60)

    board.reset()
    with analysis_lock:
        eval_history.clear()
    # Reset clock state
    turn_start_time = time.time()
    white_total_time = 0.0
    black_total_time = 0.0
    last_move_time = None
    move_time_history.clear()

    photo_tracker = BoardStateTracker()
    last_known_grid = photo_tracker.get_vision_grid()
    trigger_analysis()

    ref_board = chess.Board()
    test_sequence = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"]

    passed_count = 0

    for idx, move_san in enumerate(test_sequence, 1):
        move_obj = ref_board.parse_san(move_san)
        ref_board.push(move_obj)
        simulated_grid = board_to_grid(ref_board)

        diff_sqs = diff_grids(last_known_grid, simulated_grid)
        inferred_move, status, candidates = infer_move(board, diff_sqs)

        if inferred_move == move_obj:
            passed_count += 1
            san = board.san(inferred_move)

            # Task 11: Record move time
            elapsed = time.time() - turn_start_time
            is_white_move = board.turn == chess.WHITE
            if is_white_move:
                white_total_time += elapsed
            else:
                black_total_time += elapsed
            last_move_time = elapsed
            move_time_history.append((board.ply(), elapsed))
            turn_start_time = time.time()

            board.push(inferred_move)
            photo_tracker.apply_move(inferred_move)
            last_known_grid = simulated_grid
            trigger_analysis(san, move_time=elapsed)
            print(f"  [Simulated Move {idx}] {move_san:<6} -> INFERRED: {san:<6} | Result: PASS")
        else:
            print(f"  [Simulated Move {idx}] {move_san:<6} -> INFERRED: {inferred_move} | Result: FAIL")

    print(f"\nIntegration Test Summary: {passed_count} / {len(test_sequence)} moves passed.")
    print("GUI board, Stockfish eval bar, AR arrow, Ghost Piece & LLM Voice Coach triggered!\n" + "=" * 60)


def main():
    global show_ghost_piece, show_eval_history, show_blunder_alerts, show_move_clock
    global turn_start_time, white_total_time, black_total_time, last_move_time
    selected_square = None
    running = True

    trigger_analysis()  # Initial Stockfish evaluation

    print("\n" + "=" * 60)
    print("AI CHESS VISION COACH - COMPLETE SYSTEM & MOVE-TIME TRACKING")
    print("=" * 60)
    print("Controls:")
    print("  - Mouse Click: Select piece and destination square to move")
    print("  - 'g' Key: Toggle AR Ghost Piece floating overlay ON/OFF")
    print("  - 'h' Key: Toggle Eval Trajectory Overlay graph ON/OFF")
    print("  - 'b' Key: Toggle Blunder Alerts ON/OFF")
    print("  - 'c' Key: Toggle Chess Clock overlay ON/OFF")
    print("  - 'i' Key: Calibrate Photo Move Mode using an EMPTY board photo")
    print("  - 'm' Key: Submit a NEW board photo to detect and play a move")
    print("  - 't' Key: Run simulated end-to-end integration test")
    print("  - 'r' Key: Reset board position to starting game state")
    print("=" * 60 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    show_ghost_piece = not show_ghost_piece
                    status_str = "ENABLED" if show_ghost_piece else "DISABLED"
                    print(f"[NOTICE] Floating Ghost Piece Overlay: {status_str}")

                elif event.key == pygame.K_h:
                    show_eval_history = not show_eval_history
                    status_str = "ENABLED" if show_eval_history else "DISABLED"
                    print(f"[NOTICE] Eval Trajectory Overlay: {status_str}")

                elif event.key == pygame.K_b:
                    show_blunder_alerts = not show_blunder_alerts
                    status_str = "ENABLED" if show_blunder_alerts else "DISABLED"
                    print(f"[NOTICE] Blunder Alerts: {status_str}")

                elif event.key == pygame.K_c:
                    show_move_clock = not show_move_clock
                    status_str = "ENABLED" if show_move_clock else "DISABLED"
                    print(f"[NOTICE] Chess Clock Overlay: {status_str}")

                elif event.key == pygame.K_i:
                    initialize_photo_mode()

                elif event.key == pygame.K_m:
                    process_photo_move()

                elif event.key == pygame.K_t:
                    run_simulated_integration_test()

                elif event.key == pygame.K_r:
                    board.reset()
                    with analysis_lock:
                        eval_history.clear()
                        latest_blunder_event = None
                    # Reset clock state
                    turn_start_time = time.time()
                    white_total_time = 0.0
                    black_total_time = 0.0
                    last_move_time = None
                    move_time_history.clear()
                    if photo_tracker:
                        photo_tracker.board.reset()
                        last_known_grid = photo_tracker.get_vision_grid()
                    trigger_analysis()
                    print("[NOTICE] Reset board position to starting position.")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = pixel_to_square(pygame.mouse.get_pos())
                if clicked_square is None:
                    continue

                if selected_square is None:
                    if board.piece_at(clicked_square) is not None:
                        selected_square = clicked_square
                else:
                    move = chess.Move(selected_square, clicked_square)
                    promo_move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
                    final_move = None
                    if move in board.legal_moves:
                        final_move = move
                    elif promo_move in board.legal_moves:
                        final_move = promo_move

                    if final_move is not None:
                        san = board.san(final_move)

                        # Task 11: Record move time before applying
                        elapsed = time.time() - turn_start_time
                        is_white_move = board.turn == chess.WHITE
                        if is_white_move:
                            white_total_time += elapsed
                        else:
                            black_total_time += elapsed
                        last_move_time = elapsed
                        move_time_history.append((board.ply(), elapsed))
                        turn_start_time = time.time()

                        board.push(final_move)
                        if photo_tracker:
                            photo_tracker.apply_move(final_move)
                            last_known_grid = photo_tracker.get_vision_grid()
                        print(f"Move: {san}  |  FEN: {board.fen()}  |  Time: {elapsed:.1f}s")
                        trigger_analysis(san, move_time=elapsed)

                    selected_square = None

        draw_board(selected_square)
        draw_pieces()
        draw_best_move_arrow()
        draw_ghost_piece()
        draw_move_clock()
        draw_eval_history()
        draw_blunder_alert()
        draw_eval_bar()
        draw_commentary_panel()
        pygame.display.flip()
        clock.tick(30)

        if board.is_game_over():
            print("Game over:", board.result())

    engine.quit()
    pygame.quit()


if __name__ == "__main__":
    main()