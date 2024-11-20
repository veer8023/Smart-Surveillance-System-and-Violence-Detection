from flask import Flask, Response, request, jsonify
import cv2

app = Flask(__name__)

# Initialize variables
camera = None  # Camera instance
count = 0  # Counter for consecutive "no motion"
ALLOWED_IP = "192.168.29.125"  # Replace with the actual IP address of the RPI

# Function to start the camera
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # For USB camera or CSI camera, use /dev/video0 or 1
        print("Camera started.")

# Function to stop the camera
def stop_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        print("Camera stopped.")

# Start the camera initially
start_camera()

# Function to generate video frames
def generate_frames():
    global camera
    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to display in the HTML page
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Video streaming route, generates frames from the camera
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Main page with video feed embedded
    return '''
    <html>
    <head>
        <title>Camera Stream</title>
    </head>
    <body>
        <h1>Live Camera Stream</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''

@app.route('/data', methods=['POST'])
def receive_data():
    global count, camera
    # Check the client's IP address
    client_ip = request.remote_addr
    if client_ip != ALLOWED_IP:
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    data = request.json  # Expecting JSON data
    print(f"Received data: {data} from {client_ip}")  # Print received data

    motion = data['motion']  # Get motion status (0 or 1)

    if motion:
        count = 0  # Reset counter if motion is detected
        if camera is None or not camera.isOpened():
            start_camera()  # Turn the camera back on
    else:
        count += 1
        if count == 3:
            stop_camera()  # Stop the camera after 3 consecutive "no motion"

    return jsonify({"status": "success", "message": "Data received"}), 200

if __name__ == '__main__':
    # Start the server on Jetson's IP address (use 0.0.0.0 to be accessible on the network)
    app.run(host='0.0.0.0', port=5000)
