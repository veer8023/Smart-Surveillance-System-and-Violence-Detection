import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the pre-trained i3d model for activity recognition from TensorFlow Hub
model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
model = hub.load(model_url)

# Get the prediction function from the model's signature
model_fn = model.signatures['default']

# Initialize video capture from webcam (or video file)
cap = cv2.VideoCapture(0)  # Use '0' for the webcam or replace with a file path

# Function to preprocess video frames for model input
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize the frame to 224x224
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Class labels for the i3d model (from Kinetics-400 dataset)
LABELS = [
    "running", "walking", "jumping", "dancing", "climbing", "sitting", "falling", "swimming",
    "riding a bike", "playing a sport"
]

# Loop for real-time video capture and activity detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and get model predictions
    input_tensor = preprocess_frame(frame)
    predictions = model_fn(tf.convert_to_tensor(input_tensor))

    # Extract the prediction scores and get the class with the highest score
    predicted_label = LABELS[np.argmax(predictions['default'])]  # Get the activity label

    # Display the predicted activity on the frame
    cv2.putText(frame, f"Activity: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Activity Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
