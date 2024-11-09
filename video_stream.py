from flask import Flask, Response
import cv2

app = Flask(__name__)

# Initialize camera (use the correct index or video source if not using default)
camera = cv2.VideoCapture(0)  # For USB camera or CSI camera, use /dev/video0 or 1

def generate_frames():
    while True:
        # Capture frame-by-frame
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

if __name__ == '__main__':
    # Start the server on Jetson's IP address (use 0.0.0.0 to be accessible on the network)
    app.run(host='0.0.0.0', port=5000)
