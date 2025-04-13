"""
Extracts frames from a video file and saves them as image files in an output directory.

Dependencies
------------
- OpenCV
- OS

Parameters
----------
video_path : str
    Path to the input video file.
output_dir : str
    Path to the directory where the extracted frames will be saved.
frame_rate : int
    The interval of frames to skip before saving (default is 30).

Flow
----
- opens a video file using OpenCV
- iterates through frames sequentially
- saves every Nth frame (where N is set by `frame_rate`) as a `.jpg` file
- saves all extracted frames to the `output_dir` folder

Variables
---------
- `video_path`: Path to the input video.
- `output_dir`: Directory where extracted frames will be saved.
- `frame_rate`: Determines the interval at which frames are saved.
- `count`: Tracks the current frame number.
- `saved`: Counts and indexes the saved frames.

Returns
------
None
    Saves image files named `frame_00000.jpg`, `frame_00001.jpg`, ..., in `output_dir`.
"""

import cv2
import os

video_path = 'E:/facultate/licenta/pda/videos/challenge_color_848x480.mp4'
output_dir = '/video_output'
os.makedirs(output_dir, exist_ok=True)

video_capture = cv2.VideoCapture(video_path)
frame_rate = 30
count = 0
saved = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    if count % frame_rate == 0:
        filename = os.path.join(output_dir, f'frame_{saved:05d}.jpg')
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

video_capture.release()
print(f"Saved {saved} frames to {output_dir}")
