from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def gen_frames():
    """Generate video frames for streaming"""
    camera = cv2.VideoCapture(1)  # Use the default camera (index 0)
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
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





