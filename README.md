Railway Track Surveillance System

Project Overview

The Railway Track Surveillance System is an advanced real-time monitoring solution designed to ensure the safety and maintenance of railway tracks. The system utilizes Python, Flask, OpenCV, and YOLOv4 to detect anomalies and objects in the railway track footage. By streaming live video via a web interface, the system allows operators to monitor railway conditions and quickly identify any potential threats or maintenance issues. With motion and object detection capabilities, this surveillance system enhances the safety of the railway infrastructure.

Features

Real-Time Video Streaming: Continuously streams live footage from a webcam to a web interface for constant monitoring.
Motion and Object Detection: Powered by the YOLOv4 (You Only Look Once) deep learning model, the system detects objects such as pedestrians, vehicles, or other anomalies on the tracks.
Live Feed Interface: Provides users with an intuitive web interface to monitor the real-time video feed.
Adjustable Alarm Mode: Users can toggle motion detection on or off via the web interface to focus on specific areas of interest or reduce false alerts.
System Status Display: The web interface also shows system health information, such as camera status and object detection activity.
Customizable Thresholds: Users can adjust the sensitivity of the object detection model to suit different operational conditions (e.g., weather or lighting).
Tech Stack

Python: The primary programming language for backend processing and motion detection logic.
Flask: Lightweight web framework to serve the live video feed and user interface.
OpenCV: Open Source Computer Vision Library for video processing, including webcam feed management and real-time video streaming.
YOLOv4: Real-time object detection algorithm for robust, fast, and accurate identification of objects in the video stream.
HTML/CSS/JavaScript: Frontend technologies for the user interface, allowing users to interact with the live feed and control settings.
Installation

Prerequisites
To set up this project, ensure you have the following installed:

Python 3.x
A webcam (external or built-in)
A system capable of running YOLOv4 (GPU preferred for enhanced performance, but CPU support is also available with adjustments)
pip for managing Python dependencies
Steps
Clone the Repository
Start by cloning the repository to your local machine:

git clone https://github.com/G-aura-v/Railway-track-surveillance-system.git


Navigate to the Project Directory

Change to the project directory:

cd Railway-track-surveillance-system


Install Dependencies
Install the necessary Python packages listed in the requirements.txt file:

pip install -r requirements.txt
Download YOLOv4 Weights and Configuration Files
Download the following files for YOLOv4 from the official YOLO website or its official GitHub repository:

yolov4.weights (Pre-trained model weights)
yolov4.cfg (Configuration file for YOLOv4)
coco.names (Class names file for YOLOv4)
Place these files in the root directory of your project (the same location as app.py).
Configure Webcam
Make sure that your webcam is properly connected and detected by your system. The system will use OpenCV to access the webcam and stream the video feed for object detection.
Running the Application
Start the Surveillance System
To launch the system, run the following command:

python mypro.py
This will start the Flask server and initialize the webcam for real-time video processing and surveillance.
Access the Web Interface
Open a web browser and go to the following address to view the live video feed:

http://localhost:5000
The web interface will show the video feed, with detected objects and anomalies highlighted in the stream. The interface will also provide system health and motion detection status.
Enable/Disable Motion Detection
The motion detection alarm can be toggled using the controls on the web interface. When enabled, the system will display detected objects and track movements on the railway tracks. This can be useful for monitoring abnormal activity.
Adjust Sensitivity (Optional)
If needed, you can adjust the sensitivity or detection thresholds of the YOLOv4 model by modifying the mypro.py file (the backend logic for object detection). This can help reduce false positives or ensure accurate detection in various conditions (e.g., night time, fog, etc.).
