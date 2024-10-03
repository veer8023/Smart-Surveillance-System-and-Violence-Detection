import os
import warnings
import tensorflow as tf

# Suppress oneDNN custom operations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress all Python deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


import cv2
import numpy as np
import tensorflow_hub as hub

# Load TensorFlow Hub model (SSD MobileNet V2)
print("[INFO] Loading TensorFlow Hub model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Function to perform object detection
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections

# Initialize webcam
cap = cv2.VideoCapture(0)

# Labels for harmful objects (modify this list if you use a different dataset)
WEAPON_LABELS = ["knife", "gun"]

# Define the labels for the COCO dataset (make sure it includes harmful objects)
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",  # Include "knife"
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "hot dog", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "gun"  # Include "gun"
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break

    resized_frame = cv2.resize(frame, (640, 640))

    # Perform object detection
    detections = detect_objects(resized_frame)

    # Extract detection information
    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(np.int32)

    # Set detection confidence threshold
    threshold = 0.5

    # Initialize person count
    person_count = 0

    # Loop over detected objects
    for i in range(len(detection_scores)):
        if detection_scores[i] > threshold:
            (startY, startX, endY, endX) = detection_boxes[i]
            (h, w) = frame.shape[:2]
            (startX, startY, endX, endY) = (int(startX * w), int(startY * h),
                                            int(endX * w), int(endY * h))

            class_index = detection_classes[i] - 1
            if 0 <= class_index < len(LABELS):
                label = LABELS[class_index]
                confidence = detection_scores[i]

                # Count the number of people detected
                if label == "person":
                    person_count += 1
                    # Draw bounding box for people in green
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # Green box for people
                    text = f"Person: {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                # Focus on harmful objects (knife, gun, etc.)
                if label in WEAPON_LABELS:
                    # Draw bounding box and label for weapon detection
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)  # Red for danger
                    text = f"Weapon ({label}): {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

    # Display the count of people detected
    cv2.putText(frame, f"People Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Weapon Detection and People Counting", frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
