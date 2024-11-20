import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from flask import Flask, Response, request, jsonify
from datetime import datetime
from pathlib import Path
from numpy import random
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
from keras.models import load_model
from collections import deque
from PIL import Image

app = Flask(__name__)

# Global variables
camera = None
count = 0
ALLOWED_IP = "192.168.29.125"

# Initialize detection models
def initialize_models():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    global opt
    opt = parser.parse_args([])

    # Setup device
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load pose estimation model
    pose_model = torch.load('yolov7-w6-pose.pt', map_location=device)['model'].float().eval()
    pose_model = pose_model.to(device)

    # Load violence detection YOLO model
    vio_model = attempt_load('yolov7_vio_detect_v3.pt', map_location=device)
    vio_model.to(device).eval()

    # Load Keras violence detection model
    keras_model = load_model("modelnew.h5")
    
    return device, half, pose_model, vio_model, keras_model

# Initialize models at startup
DEVICE, HALF, POSE_MODEL, VIO_MODEL, KERAS_MODEL = initialize_models()
Q = deque(maxlen=128)  # Queue for Keras model prediction averaging

# Camera functions
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        print("Camera started.")

def stop_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        print("Camera stopped.")

# Detection functions
def run_pose_estimation(tensor_image, model):
    tensor_image = tensor_image.half() if next(model.parameters()).dtype == torch.float16 else tensor_image.float()
    pred = model(tensor_image)
    pred = pred[0] if isinstance(pred, tuple) else pred
    return non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

def run_vio_detection(img, model, device, half=False):
    img = img.to(device)
    img = img.half() if half else img.float()  
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  
        with torch.amp.autocast('cuda', enabled=half):
            pred = model(img)[0]

    pred = non_max_suppression(pred, 0.6, opt.iou_thres)
    return pred

def run_keras_detection(frame, model, Q):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255

    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    prob = results[0]
    return prob > 0.40, prob

# Video stream generation
def generate_frames():
    global camera, Q, DEVICE, HALF, POSE_MODEL, VIO_MODEL, KERAS_MODEL
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        if camera is None or not camera.isOpened():
            break
        
        success, frame = camera.read()
        if not success:
            break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Prepare frame for detection
        img = torch.from_numpy(frame).to(DEVICE)
        
        # Run violence detection
        vio_pred = run_vio_detection(img, VIO_MODEL, DEVICE, half=HALF)
        
        # Run Keras violence detection
        violence_detected, prob = run_keras_detection(frame, KERAS_MODEL, Q)
        
        # Annotate frame
        text_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        fps_text = f'FPS: {int(fps)}'
        text = f"Violence: {'Yes' if violence_detected else 'No'} | Prob: {prob:.2f}"
        cv2.putText(frame, text + " | " + fps_text, (35, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Violence Detection Camera Stream</title>
    </head>
    <body>
        <h1>Live Violence Detection Stream</h1>
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

    data = request.json
    print(f"Received data: {data} from {client_ip}")

    motion = data['motion']

    if motion:
        count = 0
        if camera is None or not camera.isOpened():
            start_camera()
    else:
        count += 1
        if count == 3:
            stop_camera()

    return jsonify({"status": "success", "message": "Data received"}), 200

if __name__ == '__main__':
    # Ensure camera is stopped at startup
    stop_camera()
    
    # Start the server
    app.run(host='0.0.0.0', port=5000)