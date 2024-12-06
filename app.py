from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np

app = Flask(__name__)

# Load the YOLOv5n model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()  # Set the model to evaluation mode

def gen_frames():
    """Generate video frames for streaming with YOLOv5 detection"""
    camera = cv2.VideoCapture(1)  # Use external camera (change index if needed)
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to RGB (YOLO expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection with YOLOv5
        results = model(rgb_frame)

        # Convert results to numpy array and draw bounding boxes
        detected_frame = np.squeeze(results.render())  # Get the rendered frame with detections

        # Convert the frame back to BGR for OpenCV
        detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', detected_frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera when done
    camera.release()

@app.route('/')
def index():
    """Serve the main page with the video feed"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
