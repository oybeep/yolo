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
        
        # 1. 여기서 conf 옵션을 0.25 정도로 넣어줍니다 (너무 낮으면 또 폭주함)
        results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)
        
        # 2. ★★★ 박스를 직접 그려서, 1.00 라벨들을 다 지워버립니다! ★★★
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 좌표 가져오기
                x1, y1, x2, y2 = box.xyxy[0]
                
                # 라벨 가져오기 (이름과 확률은 뺍니다!)
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # ★★★ 여기서 특정 클래스(예: cafe, water_dispenser)는 무시하게 만들 수도 있습니다! ★★★
                if class_name in ['self_service_cafe', 'water_dispenser']:
                    continue # 카페 관련 박스는 아예 그리지 않음
                
                # 직접 그리기 (확률 수치는 안 쓰고 이름만 쓰거나, 박스만 그립니다)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.putText(frame, f'{class_name}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 박스가 그려진 원본 frame을 인코딩해서 전송
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
