import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

model = YOLO('/home/team-d/obstacle_detection/best_incomplete_ncnn_model', task='detect')

def generate_frames():
    cap = cv2.VideoCapture(0)

    count = 0 
    annotated_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        count += 1

        if count % 3 == 0:
            results = model.predict(frame, imgsz=160, conf=0.5, verbose=False)

            annotated_frame = results[0].plot(line_width=2, labels=True, boxes=True)

        if annotated_frame is None:
            display_frame = frame
        else:
            display_frame = annotated_frame

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<body style="margin:0; background:#000;"><img src="/video_feed" style="width:100%; height:auto;"></body>'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
