Railway Track Surveillance System

Overview

The Railway Track Surveillance System monitors railway tracks in real-time to ensure safety and maintenance. Using Python, Flask, OpenCV, and YOLOv4 for object detection, the system streams live footage, detects anomalies, and displays them through a web interface. The system allows users to monitor the tracks for potential hazards and adjust motion detection settings.

Features

Real-Time Video Streaming: Provides continuous video feed from the webcam, accessible through a web interface.
Motion and Object Detection: Utilizes YOLOv4 for fast and accurate object detection to identify potential hazards on the tracks.
Live Feed Interface: A user-friendly web interface displays the live feed along with detected objects.
Adjustable Alarm Mode: Motion detection can be toggled on/off based on user preferences, with detected objects highlighted.
System Health Display: Provides information about system status and object detection activity.
Easy Configuration: Simple setup with clear instructions to get the system running on your machine.
Tech Stack

Python 3.x: Backend processing and video stream handling.
Flask: Web framework to serve the live video feed.
OpenCV: Computer vision library for handling webcam feed and video processing.
YOLOv4: Object detection model for real-time anomaly detection.
Installation

Prerequisites
Python 3.x installed.
Webcam (external or built-in) for video feed.
YOLOv4 Weights and configuration files (downloadable from the official YOLO website).
pip for managing dependencies.
Steps
Clone the Repository
Clone the project repository to your local machine:

git clone https://github.com/G-aura-v/Railway-track-surveillance-system.git
Navigate to Project Directory
Move into the project folder:

cd Railway-track-surveillance-system
Install Dependencies
Install required Python libraries:

pip install -r requirements.txt
Download YOLOv4 Files
Download the following files and place them in the root directory:

yolov4.weights (Pre-trained weights)
yolov4.cfg (Configuration file)
coco.names (Class names file)
These files can be found on the YOLO website.
Run the Application
Start the application by running:

python mypro.py
Access the Web Interface
Open a browser and navigate to:

http://localhost:5000
You will be able to view the live video feed with detected objects highlighted on the screen.
Usage

Live Feed: View the continuous video stream of the railway tracks.
Motion Detection: Enable or disable motion detection via the web interface.
Adjust Sensitivity: Modify the sensitivity of object detection by adjusting the mypro.py file for different operational conditions (e.g., lighting, weather).
System Health: The web interface shows the current status of the system, including motion detection and object detection activity.
License

MIT License

