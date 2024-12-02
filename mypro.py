import threading
import os
import cv2
import numpy as np
import imutils

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load YOLOv3
try:
    net = cv2.dnn.readNet("/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/yolov4.weights",
                          "/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/yolov4.cfg")  # Replace with paths to your weights and cfg
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open("/Users/gauravkumar/Downloads/Railway-Track-Surveillance-System/coco.names", "r") as f:  # Replace with the path to coco.names
        classes = [line.strip() for line in f.readlines()]

    print("YOLOv3 loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv3: {e}")
    cap.release()
    exit()

# Test YOLOv3 functionality with a dummy frame
ret, test_frame = cap.read()
if ret:
    test_blob = cv2.dnn.blobFromImage(test_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(test_blob)
    try:
        _ = net.forward(output_layers)  # Test if YOLO forward pass works
        print("YOLOv3 is working!")
    except Exception as e:
        print(f"Error during YOLO forward pass: {e}")
        cap.release()
        exit()
else:
    print("Error: Unable to capture video for testing YOLOv3.")
    cap.release()
    exit()

# Initialize motion detection variables
ret, start_frame = cap.read()
if not ret:
    print("Error: Unable to capture video.")
    cap.release()
    exit()

start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm_mode = False
motion_detected = False
alarm_playing = False

# Function to play an alarm sound
def beep_alarm():
    os.system('afplay "/Users/gauravkumar/Downloads/yt1s.com - Wrong Buzzer  Sound Effect.mp3"')  # Replace with the alarm sound file path

# Function to detect objects using YOLOv3
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
            if confidence > 0.5:  # Confidence threshold
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

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    frame = imutils.resize(frame, width=500)

    if alarm_mode:
        # Convert the frame to grayscale and blur it
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (21, 21), 0)

        # Compute the absolute difference between the current frame and the starting frame
        difference = cv2.absdiff(frame_bw, start_frame)
        _, threshold = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

        # Detect motion
        motion_detected = cv2.countNonZero(threshold) > 5000  # Sensitivity threshold

        if motion_detected:
            if not alarm_playing:
                alarm_playing = True
                threading.Thread(target=beep_alarm).start()

            # Display "Motion Detected" on the frame
            cv2.putText(frame, "Motion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detect objects using YOLOv3
            frame, detected_objects = detect_objects(frame)

            if detected_objects:
                print(f"Objects detected: {detected_objects}")
            alarm_playing = False
        else:
            cv2.putText(frame, "No Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('a'):  # Press 'a' to toggle alarm mode
        alarm_mode = not alarm_mode
        print("Alarm mode:", "ON" if alarm_mode else "OFF")

# Release resources
cap.release()
cv2.destroyAllWindows()
