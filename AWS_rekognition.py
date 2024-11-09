import boto3
import cv2
import time

# AWS credentials (replace with your actual keys)
access_key = ""
secret_key = ""

# Initialize AWS Rekognition client
rekognition_client = boto3.client(
    'rekognition',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='ap-south-1'  # Use your preferred region
)

# Configure camera and frame rate
cap = cv2.VideoCapture(0)  # Use the first connected camera
frame_interval = 5  # Capture a frame every 5 seconds
last_capture_time = time.time()

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera")
        break

    # Check if enough time has passed to capture the next frame
    current_time = time.time()
    if current_time - last_capture_time >= frame_interval:
        # Resize frame to reduce data size for faster processing
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Encode the frame as JPEG
        ret, jpeg_frame = cv2.imencode('.jpg', resized_frame)
        if not ret:
            print("Failed to encode frame")
            continue
        
        # Update the time of the last captured frame
        last_capture_time = current_time
        
        # Send frame to Rekognition for object detection
        try:
            response = rekognition_client.detect_labels(
                Image={'Bytes': jpeg_frame.tobytes()},
                MaxLabels=10,
                MinConfidence=70  # Confidence threshold for filtering
            )
            
            # Print detected objects with their confidence levels
            print("Detected objects:")
            for label in response['Labels']:
                print(f"{label['Name']} - {label['Confidence']:.2f}%")
                # Display label and confidence on the frame
                cv2.putText(frame, f"{label['Name']} ({label['Confidence']:.2f}%)",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        except Exception as e:
            print(f"Error during object detection: {e}")
    
    # Display the frame with detected labels
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
