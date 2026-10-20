"""
Quick utility: check which camera indexes are available on your system.
Run this once to figure out which number corresponds to your phone camera
vs your laptop's built-in camera.
"""

import cv2

print("Checking camera indexes 0 through 4...")

for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"  Index {index}: WORKING - resolution {width}x{height}")
        else:
            print(f"  Index {index}: opened but couldn't read a frame")
        cap.release()
    else:
        print(f"  Index {index}: not available")

print("\nUse whichever index showed 'WORKING' with the resolution matching your phone.")