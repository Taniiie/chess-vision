import sys
sys.path.append('.')
import os
import cv2
import numpy as np
from extract_video_frames import extract_frames_automatically, extract_frames_manually

def generate_test_gameplay_video(filename="scratch/test_gameplay.mp4"):
    os.makedirs("scratch", exist_ok=True)
    fps = 30
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Base board canvas
    board = np.zeros((height, width, 3), dtype=np.uint8) + 180
    cv2.rectangle(board, (120, 40), (520, 440), (60, 100, 150), -1)

    # 1. Steady empty board pause (0.0s to 1.5s -> 45 frames)
    for _ in range(45):
        frame = board.copy()
        # Add tiny noise
        noise = np.random.normal(0, 1, frame.shape).astype(np.uint8)
        out.write(cv2.add(frame, noise))

    # 2. Hand motion (1.5s to 2.5s -> 30 frames)
    for i in range(30):
        frame = board.copy()
        # Hand moving across screen
        x_pos = 100 + i * 12
        cv2.circle(frame, (x_pos, 240), 40, (40, 40, 200), -1)
        out.write(frame)

    # 3. Steady board after move 1 (2.5s to 4.0s -> 45 frames)
    board1 = board.copy()
    cv2.circle(board1, (240, 240), 20, (255, 255, 255), -1) # Move 1 piece placed
    for _ in range(45):
        frame = board1.copy()
        noise = np.random.normal(0, 1, frame.shape).astype(np.uint8)
        out.write(cv2.add(frame, noise))

    # 4. Hand motion (4.0s to 4.8s -> 24 frames)
    for i in range(24):
        frame = board1.copy()
        x_pos = 120 + i * 15
        cv2.circle(frame, (x_pos, 180), 40, (40, 40, 200), -1)
        out.write(frame)

    # 5. Steady board after move 2 (4.8s to 6.0s -> 36 frames)
    board2 = board1.copy()
    cv2.circle(board2, (360, 240), 20, (20, 20, 20), -1) # Move 2 piece placed
    for _ in range(36):
        frame = board2.copy()
        noise = np.random.normal(0, 1, frame.shape).astype(np.uint8)
        out.write(cv2.add(frame, noise))

    out.release()
    print(f"Generated synthetic test video: '{filename}'")
    return filename

def test_video_extraction():
    video_file = generate_test_gameplay_video()

    print("\n--- Testing Automatic Steady-Frame Extraction ---")
    auto_meta = extract_frames_automatically(video_file, output_dir="scratch/test_extracted_frames")

    print("\n--- Testing Manual Timestamp Override Extraction ---")
    manual_meta = extract_frames_manually(video_file, [0.8, 3.2, 5.4], output_dir="scratch/test_manual_frames")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_video_extraction()
