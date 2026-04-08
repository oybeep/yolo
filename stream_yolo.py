import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)
# 사용자님이 만든 NCNN 모델 경로를 정확히 적어주세요
model = YOLO('/home/team-d/obstacle_detection/best_incomplete_ncnn_model', task='detect')

def generate_frames():
    # 카메라 연결 (0번 또는 1번)
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(frame, imgsz=320, verbose=False)
            
            annotated_frame = results[0].plot()
            
            # 화면을 JPEG로 압축해서 전송 준비
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            # 웹 브라우저로 한 프레임씩 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    # 웹 주소 접속 시 실시간 영상 스트리밍 시작
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 라즈베리파이의 IP로 서버 실행 (포트 5000)
    app.run(host='172.20.10.3', port=5000)
