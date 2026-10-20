# ChessVision: AI Computer Vision Chess Coach and AR Interface

ChessVision is an end-to-end intelligent chess companion that bridges physical chessboard play and computational chess intelligence. Using computer vision, physical board photos or camera feeds are captured, rectified, and tracked to maintain a synchronized digital game state. The system features an interactive graphical user interface, real-time Stockfish engine evaluation, augmented reality move guidance, and an LLM-driven voice coach providing auditory commentary on every move.

---

## Current Project Status

Note: This project is actively under development and represents a work in progress. Core computer vision stages, engine integration, graphical rendering, and voice coaching are operational, while advanced continuous video tracking and deep learning piece recognition are planned for future releases.

---

## System Architecture

The ChessVision architecture consists of three interconnected subsystems:

1. Computer Vision Pipeline: Detects the physical board, normalizes geometry via perspective transformation, determines square occupancy, classifies piece colors, and deduces legal chess moves.
2. Interactive AR Graphical Interface: Renders the digital board, displays real-time Stockfish evaluation bars, plots score trajectories, and projects floating ghost piece guidance for recommended moves.
3. AI Voice Coach: Leverages high-throughput LLMs via the Groq API (Llama 3.1) to generate insightful natural language commentary, synthesized asynchronously through an offline text-to-speech engine.

---

## Completed Work

### 1. Computer Vision Pipeline
- Chessboard Corner Detection: Multi-stage detection using cv2.findChessboardCornersSB, adaptive thresholding, CLAHE contrast enhancement, and bilateral filtering fallback routines (board_detection.py, corners.py, static_board_detection.py).
- Perspective Rectification: Outer corner extrapolation and homography transformation mapping raw board images into canonical 800x800 pixel top-down representations (perspective_warp.py).
- Square Segmentation: Grid partitioning dividing rectified boards into 64 individual square crops with centered margin cropping to avoid grid line noise (split_squares.py).
- Occupancy Analysis: Variance-based and contrast-based thresholding algorithms to accurately classify occupied versus empty squares across standard starting positions and open gameplay (occupancy.py).
- Piece Color Identification: Luminance and color distribution analysis distinguishing White pieces, Black pieces, and unoccupied squares (color_detection.py, piece_color.py).
- Synthetic-Trained Piece Classification: Procedural generation of synthetic piece silhouette training data and k-Nearest Neighbors (k-NN) classification pipeline (train_synthetic_classifier.py, piece_type.py).
- Move Inference and State Tracking: BoardStateTracker tracking board deltas between successive frames, translating occupancy differences into valid python-chess moves including captures and basic piece displacement (move_tracker.py, step6g_move_tracker.py).

### 2. Interactive GUI and Augmented Reality Display
- Pygame Game Interface: Complete mouse-driven interactive chessboard supporting piece dragging, legal move validation, and board state synchronization (chess_gui.py).
- Real-Time Stockfish Engine Integration: Background UCI engine communication evaluating positions with dynamic centipawn and mate scoring.
- Visual Evaluation Bar: Smoothly updating vertical evaluation indicator displaying positional advantage for White and Black.
- Best-Move Visual Guidance: Directional vector arrow rendering the engine-recommended candidate move directly on the board.
- AR Ghost Piece Overlay: Translucent, hovering piece visualization that demonstrates the destination square of the engine's recommended move (toggled with the 'g' key).
- Evaluation Trajectory Graph: Live plot illustrating centipawn swings over the progression of plies (toggled with the 'h' key).
- Blunder and Brilliant Move Detection: Centipawn delta tracking alerting users to major blunders (-150 cp) or brilliant tactics (+150 cp).
- Move-Time Tracking: Turn duration timing identifying time trouble or prolonged calculation phases.
- Photo Move Mode: Capability to calibrate from an empty board image ('i' key) and apply physical board moves from photo captures ('m' key).

### 3. LLM Voice Coach and Audio Commentary
- Groq Cloud API Integration: Ultra-fast LLM inference using llama-3.1-8b-instant with fallback handling to alternative models (step7_commentary.py).
- Context-Aware Coaching: Prompts incorporate current board FEN, last played move, evaluation swings, blunders, and calculation times to produce punchy 1-2 sentence coaching insights.
- Response Caching: In-memory caching layer preventing duplicate API invocations for recurring positions.
- Non-Blocking Text-to-Speech: Dedicated worker thread utilizing pyttsx3, ensuring voice synthesis does not interrupt the Pygame render loop.
- UI Status Indication: Real-time visual status panel indicating when the AI coach is generating or vocalizing feedback.

### 4. Utilities and Diagnostics
- Camera Hardware Verification: Device enumeration and backend configuration testing (check_cameras.py, list_camera_names.py, webcam.py).
- Video Frame Extraction: Tools for sampling frames from recorded games for offline evaluation (extract_video_frames.py).
- Automated Non-GUI Test Suite: Standalone test script validating dependencies, image assets, Stockfish connectivity, and vision algorithms (run_tests.py).

---

## Future Roadmap

The following milestones are planned for subsequent iterations:

1. Continuous Real-Time Video Move Detection:
   - Transition from manual snapshot capture ('m' key) to automated video stream tracking.
   - Hand detection and occlusion filtering using optical flow or background subtraction to detect when a player has finished their move.

2. Deep Learning Piece Recognition:
   - Replace the procedural synthetic k-NN classifier with a fine-tuned convolutional neural network (CNN) or YOLOv8 piece detector trained on varied real-world chess sets.
   - Robust classification under diverse lighting, cast shadows, and custom piece designs.

3. Perspective and Lighting Invariance:
   - Dynamic corner tracking accommodating angled camera placements and board movement.
   - Adaptive normalization to handle glare, reflections, and uneven ambient illumination.

4. Game Analysis and Export:
   - Automatic PGN export containing timestamps, engine evaluations, and AI commentary annotations.
   - Post-game summary reports highlighting key turning points and tactical mistakes.

5. Hardware and Mobile Integration:
   - Support for mobile camera streams via RTSP/HTTP.
   - Exploration of wearable AR glasses (e.g., projection directly onto physical boards).

---

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux
- Stockfish chess engine binary
- Groq Cloud API key (for AI voice commentary)

### 1. Clone the Repository
```bash
git clone https://github.com/Taniiie/chess-vision.git
cd chess-vision
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Stockfish Setup
1. Download the appropriate Stockfish binary for your operating system from the official website (https://stockfishchess.org/download/).
2. Place the executable inside the stockfish/ directory.
3. Update STOCKFISH_PATH in chess_gui.py and test_engine.py if your executable name or location differs:
   ```python
   STOCKFISH_PATH = r"stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
   ```

### 5. Configure API Key
Create a .env file in the root directory (refer to .env.example):
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the Application

### 1. Verification Test
Verify core components, engine connectivity, and computer vision routines:
```bash
python run_tests.py
```

### 2. Interactive GUI and Voice Coach
Launch the main application:
```bash
python chess_gui.py
```

### Keyboard Controls in GUI

| Key | Action |
| --- | --- |
| Mouse Left Click | Select square, drag and drop piece |
| i | Calibrate empty board reference photo |
| m | Input a new board photo to detect and apply move |
| t | Run end-to-end integration test simulation |
| g | Toggle AR Ghost Piece floating overlay |
| h | Toggle evaluation trajectory graph overlay |
| b | Toggle blunder and brilliant move alerts |
| c | Toggle chess clock and move timer display |
| r | Reset board to starting position |

### 3. Standalone Live Camera Vision
Test camera capture and live board detection:
```bash
python live_chess_vision.py
```

---

## Directory Structure

```text
ChessVision/
|-- assets/
|   \-- pieces/                  Piece sprite graphics (PNG)
|-- scratch/                     Diagnostic scripts and test video samples
|-- stockfish/                   Stockfish engine documentation and configuration
|-- training_data/               Synthetic dataset generation folders
|-- board_detection.py           Chessboard corner detection algorithms
|-- check_cameras.py             Webcam index and backend diagnostic utility
|-- check_file_type.py           File format verification utility
|-- chess_gui.py                 Main Pygame interface, AR overlay, and engine loop
|-- color_detection.py           Color thresholding and mask generation
|-- corners.py                   Corner extraction and subpixel refinement
|-- extract_video_frames.py      Frame extraction utility for video files
|-- live_chess_vision.py         Live camera stream occupancy detection
|-- move_tracker.py              Grid difference tracking and move inference
|-- occupancy.py                 Square occupancy classification
|-- perspective_warp.py          Perspective warping and homography
|-- piece_color.py               Piece color classification pipeline
|-- piece_type.py                Piece type classifier evaluation
|-- run_tests.py                 Automated test suite
|-- split_squares.py             Grid splitting and square crop extraction
|-- static_board_detection.py    Still-image board detection
|-- step6g_move_tracker.py       Integrated move tracker module
|-- step7_commentary.py          Groq LLM commentary and pyttsx3 TTS engine
|-- test_engine.py               Stockfish connection verification script
|-- train_synthetic_classifier.py Synthetic data generator and k-NN trainer
|-- webcam.py                    Simple camera feed viewer
|-- .env.example                 Environment variable template
|-- .gitignore                   Git ignore specifications
|-- requirements.txt             Python package dependencies
\-- README.md                    Project documentation
```

---

## License

This project is licensed under the MIT License. Stockfish is distributed under the GNU General Public License v3.
