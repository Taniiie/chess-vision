"""
AI Chess Vision Coach - Step 6a: Detect your real chessboard

Uses cv2.findChessboardCorners() - a function built into OpenCV
specifically for finding checkerboard patterns (normally used for
camera calibration, but perfect for our use case too).

It looks for the INTERNAL corners of the board - where 4 squares
meet. An 8x8 board has a 7x7 grid of these internal corner points.

IMPORTANT: Clear all pieces off the board for this test. Pieces
block the checkered pattern the detector needs to see.


Press 'q' to quit.
"""

import cv2

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

# For a standard 8x8 board, the internal corner grid is 7x7
CHESSBOARD_SIZE = (7, 7)

print("Point your camera at the EMPTY chessboard (no pieces). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # This does all the heavy lifting: searches the image for a 7x7
    # grid of checkerboard corners. Returns found=True/False and the
    # pixel locations of each corner if found.
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        # Draws small colored circles + connecting lines at each
        # detected corner - a visual sanity check that it worked.
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, found)
        cv2.putText(
            frame, "BOARD DETECTED", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame, "Searching for board...", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    cv2.imshow("Board Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()