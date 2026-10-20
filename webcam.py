"""
OpenCV Exercise 1: Open your webcam and display the live feed.
Press 'q' to quit.
"""

import cv2

# 0 usually means your default/built-in webcam.
# If you have multiple cameras and this shows the wrong one, try 1 or 2.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam. Try changing the index (0, 1, 2...).")
    exit()

print("Webcam opened. Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()  # ret = success flag, frame = the actual image (a numpy array)

    if not ret:
        print("Error: failed to grab frame.")
        break

    cv2.imshow("Webcam Feed", frame)

    # waitKey(1) waits 1ms for a keypress; without this, the window won't update or respond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()          # release the camera so other apps can use it
cv2.destroyAllWindows()  # close the display window