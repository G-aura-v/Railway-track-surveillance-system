from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import numpy as np
import imutils
import time
import os
from datetime import datetime

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
detected_objects = []
motion_status = "No Motion"
alarm_mode = False
last_alarm_time = 0
ALARM_COOLDOWN = 3
MOTION_SENSITIVITY = 5000

def init_camera():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load YOLO
    global net, layer_names, output_layers, classes
    net = cv2.dnn.readNet("/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/yolov4.weights",
                          "/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open("/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected

def process_frames():
    global output_frame, lock, detected_objects, motion_status, alarm_mode
    
    previous_frame = None
    
    while True:
        if camera is None or not camera.isOpened():
            continue
            
        ret, frame = camera.read()
        if not ret:
            continue

        frame = imutils.resize(frame, width=500)
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        if previous_frame is not None and alarm_mode:
            frame_delta = cv2.absdiff(previous_frame, current_frame_gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_pixels = cv2.countNonZero(thresh)
            
            if motion_pixels > MOTION_SENSITIVITY:
                motion_status = "Motion Detected!"
                cv2.putText(frame, motion_status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame, objects = detect_objects(frame)
                detected_objects = objects
            else:
                motion_status = "No Motion"
                cv2.putText(frame, motion_status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                detected_objects = []

        previous_frame = current_frame_gray

        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                   mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/status')
def get_status():
    return jsonify({
        'motion_status': motion_status,
        'detected_objects': detected_objects,
        'alarm_mode': alarm_mode
    })

@app.route('/toggle_alarm/<int:state>')
def toggle_alarm(state):
    global alarm_mode
    alarm_mode = bool(state)
    return jsonify({'alarm_mode': alarm_mode})

if __name__ == '__main__':
    init_camera()
    t = threading.Thread(target=process_frames)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)