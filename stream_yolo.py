import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

model = YOLO('/home/team-d/obstacle_detection/best_incomplete_ncnn_model', task='detect')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = model.predict(frame, imgsz=320, verbose=False, stream=True)
        
        for r in results:
            annotated_frame = r.plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
