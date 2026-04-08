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

        results = model.predict(frame, imgsz=320, conf=0.5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in ['self_service_cafe', 'water_dispenser']:
                    continue 

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<img src="/video_feed" style="width:100%; height:auto;">'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
