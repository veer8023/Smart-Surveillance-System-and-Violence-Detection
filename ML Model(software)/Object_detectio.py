import torch
import cv2
import numpy as np

# Load the YOLOv5 model (assuming you're using the small version, 'yolov5s.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# Define dangerous objects (COCO classes for weapons like gun, knife)
dangerous_objects = ['knife', 'gun']

# Start capturing video from the Raspberry Pi camera or connected camera
cap = cv2.VideoCapture(0)  # Change to the appropriate video source (0 for default camera)

while True:
    # Capture each frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    img = cv2.resize(frame, (640, 640))

    # Perform object detection
    results = model(img)

    # Convert results to Pandas DataFrame for easier processing
    detected_objects = results.pandas().xyxy[0]  # Contains detected objects information

    # Loop through detected objects and check for dangerous objects
    for index, row in detected_objects.iterrows():
        class_name = row['name']  # Get the class name of the object
        if class_name in dangerous_objects:
            # Get bounding box coordinates
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']

            # Draw bounding box and label
            label = f'{class_name} {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the result frame
    cv2.imshow('Dangerous Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
