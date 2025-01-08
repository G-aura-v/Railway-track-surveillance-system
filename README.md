# Railway Track Surveillance System

## Overview

The **Railway Track Surveillance System** is designed to monitor railway tracks in real-time for safety and maintenance. It uses **Python**, **Flask**, **OpenCV**, and **YOLOv4** for object detection, streaming live footage and detecting anomalies. The system provides a web interface for real-time monitoring and allows users to adjust the motion detection alarm settings.

## Features

- **Real-Time Video Streaming**: Continuous webcam video feed accessible through a web interface.
- **Motion and Object Detection**: Detect anomalies using **YOLOv4** for fast and accurate detection.
- **Live Feed Interface**: Displays live video with detected objects highlighted.
- **Adjustable Alarm Mode**: Toggle motion detection on or off.
- **System Health Display**: Information about system status and detection activity.
- **Simple Setup**: Easy configuration for quick deployment.

## Tech Stack

- **Python 3.x**: Backend logic and video stream handling.
- **Flask**: Web framework to serve live video feed.
- **OpenCV**: For video processing and webcam access.
- **YOLOv4**: Real-time object detection for anomaly identification.

## Installation

### Prerequisites

- **Python 3.x**
- **Webcam** (external or built-in)
- **YOLOv4 Weights** and config files (download from the official YOLO site)
- **pip** for managing dependencies

### Steps

1. Clone the Repository**

   ```bash
   git clone https://github.com/G-aura-v/Railway-track-surveillance-system.git
2. Navigate to Project Directory
- Move into the project folder:
    ```bash
    cd Railway-track-surveillance-system
3. Install Dependencies
- Install required Python libraries:
     ```bash
     pip install -r requirements.txt
4.Download YOLOv4 Files
Download the following files and place them in the root directory:

- yolov4.weights (Pre-trained weights)
- yolov4.cfg (Configuration file)
- coco.names (Class names file)
- These files can be found on the YOLO website.
- Run the Application
  
 5. Start the application by running:
      ```bash
      python mypro.py
6. Access the Web Interface
    Open a web browser and go to:
     ```bash
     http://localhost:5000

## Usage

- Live Video Feed: View the real-time video stream of the railway tracks.
- Motion and Object Detection: The YOLOv4 model will highlight detected objects and anomalies in the live video feed.
- Adjustable Motion Detection: You can enable or disable motion detection as needed via the web interface.
- Sensitivity Settings: Modify the detection sensitivity by adjusting the settings in the mypro.py file to suit the environment.
- System Health: The web interface will display the current system status, including whether motion detection and object detection are active.

## License

This project is licensed under the MIT License. For more details, check the LICENSE file in the repository.
