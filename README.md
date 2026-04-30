# Vision-Based-Pick-and-Place-Simulation-System

A real-time computer vision project that detects objects using YOLOv8 and simulates a robotic pick-and-place pipeline.

Features:
1) Real-time object detection using YOLOv8
2) Contrast enhancement using CLAHE for better visibility
3) Detection of small and distant objects using ROI-based zoom
4) Queue-based pick-and-place simulation
5) Smooth object movement to a defined drop zone
6) Simple visual interface with status and counters
7) Tech Stack
8) Python
9) OpenCV
10) Ultralytics YOLOv8
11) NumPy

Tech Stack:
1) Python
2) OpenCV
3) Ultralytics YOLOv8
4) NumPy

How it works:
1) The webcam captures live video
2) Frames are enhanced using CLAHE
3) YOLOv8 detects objects in the frame
4) Detected objects are added to a queue
5) The system simulates picking each object and moving it to a drop zone

Controls: 

Press P → Start picking detected objects

Press R → Reset system

Press ESC → Exit

Installation:
1) pip install -r requirements.txt
2) Run
3) python main.py
