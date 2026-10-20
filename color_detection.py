"""
OpenCV Exercise 2: Detect a colored object in your webcam feed.

Concept: we convert each frame from BGR (Blue-Green-Red) to HSV
(Hue-Saturation-Value). HSV is much easier to filter by color than
BGR, because "hue" directly represents the color itself, separate
from brightness/lighting. This matters a LOT for chessboard/piece
detection later, since lighting will vary.

Default range below is tuned for a RED object - hold up something
red (a red pen cap, red phone case, etc.) to test it.

Press 'q' to quit.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam.")
    exit()

# HSV range for "red" - red wraps around the hue circle (0 and 180 are
# both red), so we often need two ranges and combine them.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

print("Hold up a red object. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a "mask": a black & white image where white = pixels
    # that fall inside our red color range, black = everything else.
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours (outlines of connected white regions) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pick the largest contour - assume it's our object, ignore small noise
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:  # ignore tiny specs
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, "Red object", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Mask (what the computer sees as 'red')", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()