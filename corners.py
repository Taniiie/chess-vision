"""
OpenCV Exercise 3: Detect a rectangular object and find its 4 corners.

This is the SAME core technique we'll use to detect your physical
chessboard later: find edges -> find the outline (contour) -> simplify
that outline down to its corner points -> use those 4 corners to know
exactly where the board is in the frame.

Hold up something rectangular and flat - a book, a phone, an ID card,
a piece of paper. Works best with good contrast against the background.

Press 'q' to quit.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

print("Hold up a rectangular object (book, phone, card). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: grayscale - edge detection doesn't need color info, just brightness changes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: blur slightly - reduces noise so we don't detect tiny fake edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection - finds pixels where brightness changes sharply
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: find contours (outlines) from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    if contours:
        # Look through contours, largest first, for one that simplifies to ~4 points
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(contour) < 3000:  # skip small/noisy contours
                continue

            # approxPolyDP simplifies a wiggly outline down to key corner points.
            # Higher epsilon = more aggressive simplification (handles rounded
            # corners better, e.g. phones, but can distort sharp shapes if too high).
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            print(f"  [debug] contour area={int(cv2.contourArea(contour))}, corners found={len(approx)}")

            if len(approx) == 4:  # found a 4-cornered shape
                best_quad = approx
                break

    if best_quad is not None:
        cv2.drawContours(frame, [best_quad], -1, (0, 255, 0), 3)
        for point in best_quad:
            x, y = point[0]
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Edges (what Canny detects)", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()