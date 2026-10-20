"""
AI Chess Vision Coach - Step 6h: Extract Still Frames from Continuous Gameplay Video

Goal:
Process a continuous gameplay video recording (empty board, followed by moves with brief pauses)
to automatically detect steady/paused moments and extract clean still frames (frame_00_empty.jpg,
frame_01.jpg, frame_02.jpg, etc.) for Photo Move Mode.

Includes:
- Automatic Steady-Frame Detection (Mean Absolute Difference of consecutive frames).
- Visual Contact-Sheet Summary image (extracted_frames_contact_sheet.jpg).
- Manual Timestamp Override option (e.g. [1.5, 5.2, 10.8]).
"""

import os
import sys
import argparse
import numpy as np
import cv2


def compute_motion_metric(curr_gray, prev_gray):
    """
    PART A: Motion Metric Computation
    Computes Mean Absolute Difference (MAD) between two consecutive grayscale frames.
    
    Why MAD?
    - Extremely fast and computationally efficient.
    - Highly sensitive to hand movements and piece placements (spikes > 5.0).
    - Very low during steady camera pauses (< 1.5).
    """
    diff = cv2.absdiff(curr_gray, prev_gray)
    return float(np.mean(diff))


def extract_frames_automatically(video_path, output_dir="extracted_frames",
                                 motion_threshold=2.5, min_stable_sec=0.5):
    """
    PART A: Automatic Steady-Frame Detection
    1. Reads video frame by frame.
    2. Identifies continuous periods where motion < motion_threshold.
    3. Picks the MIDDLE frame of each stable stretch (least motion blur).
    4. Saves frames as frame_00_empty.jpg, frame_01.jpg, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_stable_frames = int(min_stable_sec * fps)

    print(f"Processing Video: '{video_path}'")
    print(f"  - FPS: {fps:.2f} | Total Frames: {total_frames} | Min Stable Duration: {min_stable_sec}s ({min_stable_frames} frames)")

    os.makedirs(output_dir, exist_ok=True)

    prev_gray = None
    frame_idx = 0
    motion_scores = []

    # 1. Compute frame-by-frame motion metrics
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample for speed
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            m = compute_motion_metric(gray, prev_gray)
            motion_scores.append(m)
        else:
            motion_scores.append(0.0)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    # 2. Cluster stable frame ranges
    stable_ranges = []
    in_stable = False
    start_f = 0

    for idx, score in enumerate(motion_scores):
        if score < motion_threshold:
            if not in_stable:
                in_stable = True
                start_f = idx
        else:
            if in_stable:
                in_stable = False
                end_f = idx - 1
                if (end_f - start_f + 1) >= min_stable_frames:
                    stable_ranges.append((start_f, end_f))

    if in_stable and (len(motion_scores) - 1 - start_f + 1) >= min_stable_frames:
        stable_ranges.append((start_f, len(motion_scores) - 1))

    # 3. Extract representative middle frame for each stable stretch
    cap = cv2.VideoCapture(video_path)
    extracted_metadata = []

    print(f"\nDetected {len(stable_ranges)} stable pause periods in video:")

    for period_idx, (s_frame, e_frame) in enumerate(stable_ranges):
        mid_frame = (s_frame + e_frame) // 2
        timestamp_sec = mid_frame / fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        if period_idx == 0:
            filename = "frame_00_empty.jpg"
        else:
            filename = f"frame_{period_idx:02d}.jpg"

        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

        metadata = {
            "filename": filename,
            "filepath": filepath,
            "frame_idx": mid_frame,
            "timestamp": timestamp_sec,
            "start_frame": s_frame,
            "end_frame": e_frame,
            "image": frame
        }
        extracted_metadata.append(metadata)

        print(f"  - [{filename}] Mid-Frame: {mid_frame} | Timestamp: {timestamp_sec:.2f}s | Range: {s_frame}-{e_frame}")

    cap.release()

    # PART B: Sanity check & Contact Sheet
    sanity_check_and_contact_sheet(extracted_metadata, output_dir)
    return extracted_metadata


def extract_frames_manually(video_path, timestamps_sec, output_dir="extracted_frames"):
    """
    PART C: Manual Override Option
    Extracts frames at explicit timestamps provided by the user in seconds (e.g. [1.5, 5.2, 10.8]).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    os.makedirs(output_dir, exist_ok=True)
    extracted_metadata = []

    print(f"Manual Override Extraction for Video: '{video_path}'")

    for idx, t_sec in enumerate(timestamps_sec):
        target_frame = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"  [Warning] Could not read frame at timestamp {t_sec:.2f}s")
            continue

        filename = "frame_00_empty.jpg" if idx == 0 else f"frame_{idx:02d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

        metadata = {
            "filename": filename,
            "filepath": filepath,
            "frame_idx": target_frame,
            "timestamp": t_sec,
            "image": frame
        }
        extracted_metadata.append(metadata)
        print(f"  - Extracted manual frame [{filename}] at {t_sec:.2f}s (frame {target_frame})")

    cap.release()
    sanity_check_and_contact_sheet(extracted_metadata, output_dir)
    return extracted_metadata


def sanity_check_and_contact_sheet(extracted_metadata, output_dir):
    """
    PART B: Sanity Check Output & Visual Contact-Sheet Grid Generation
    """
    count = len(extracted_metadata)
    print("\n" + "=" * 60)
    print("SANITY CHECK & CONTACT SHEET GENERATION")
    print("=" * 60)
    print(f"Total Still Frames Extracted: {count}")

    if count < 2:
        print("[WARNING] Very few stable frames found (< 2). The video may have constant movement,")
        print("          or motion threshold may be too strict. Consider using manual timestamps!")
    elif count > 20:
        print("[WARNING] High number of stable frames found (> 20). Motion threshold may be too sensitive.")
    else:
        print("Frame count is within reasonable expected range for game moves.")

    if count == 0:
        return

    # Build visual contact sheet grid (thumbnail tiles)
    thumb_w, thumb_h = 240, 180
    cols = min(4, count)
    rows = (count + cols - 1) // cols

    sheet = np.zeros((rows * (thumb_h + 30), cols * thumb_w, 3), dtype=np.uint8) + 30

    for i, meta in enumerate(extracted_metadata):
        r, c = i // cols, i % cols
        img = meta["image"]
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        y_off = r * (thumb_h + 30)
        x_off = c * thumb_w

        # Place thumbnail
        sheet[y_off:y_off+thumb_h, x_off:x_off+thumb_w] = thumb

        # Label banner
        label = f"{meta['filename']} ({meta['timestamp']:.1f}s)"
        cv2.rectangle(sheet, (x_off, y_off + thumb_h), (x_off + thumb_w, y_off + thumb_h + 30), (15, 15, 20), -1)
        cv2.putText(sheet, label, (x_off + 8, y_off + thumb_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    contact_path = os.path.join(output_dir, "contact_sheet.jpg")
    cv2.imwrite(contact_path, sheet)
    print(f"Visual contact sheet saved to '{contact_path}'\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract still frames from gameplay video.")
    parser.add_argument("video", nargs="?", default="gameplay.mp4", help="Path to video file (default: gameplay.mp4)")
    parser.add_argument("--timestamps", type=str, default=None, help="Comma-separated manual timestamps in seconds (e.g. 1.5,5.2,10.8)")
    parser.add_argument("--outdir", type=str, default="extracted_frames", help="Output directory for frames")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[Error] Video file '{args.video}' not found.")
        print("Usage:")
        print("  python extract_video_frames.py <video_path>")
        print("  python extract_video_frames.py <video_path> --timestamps 1.5,5.2,10.8")
        sys.exit(1)

    if args.timestamps:
        ts = [float(x.strip()) for x in args.timestamps.split(",") if x.strip()]
        extract_frames_manually(args.video, ts, output_dir=args.outdir)
    else:
        extract_frames_automatically(args.video, output_dir=args.outdir)


if __name__ == "__main__":
    main()
