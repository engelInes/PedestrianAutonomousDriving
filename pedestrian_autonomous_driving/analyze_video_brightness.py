"""
Analyzes the average brightness of frames from a video and plots the brightness over time.

Dependencies
------------
- OpenCV
- NumPy
- Matplotlib

Flow
----
- loads a video file using OpenCV.
- iterates through the video frames, sampling every Nth frame.
- converts each sampled frame to grayscale and calculates its average pixel intensity.
- for each frame number, it stores the average brightness.
- plots the average brightness across sampled frames using Matplotlib.

Notes
-----
- `matplotlib.use('TkAgg')` is specified for compatibility with some GUI environments.
- The video has 30 FPS.
"""

import matplotlib
matplotlib.use('TkAgg')

import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = "E:/facultate/licenta/pda/videos/challenge_color_848x480.mp4"
video_capture = cv2.VideoCapture(video_path)

frame_brightness = []
frame_numbers = []
frame_interval = 30

frame_count = 0
success, frame = video_capture.read()
while success:
    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        frame_brightness.append(brightness)
        frame_numbers.append(frame_count)
    success, frame = video_capture.read()
    frame_count += 1

video_capture.release()

plt.figure(figsize=(12, 5))
plt.plot(frame_numbers, frame_brightness, marker='o', linestyle='-')
plt.title("Average Frame Brightness Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Average Brightness (Grayscale Intensity)")
plt.grid(True)
plt.tight_layout()
plt.show()
