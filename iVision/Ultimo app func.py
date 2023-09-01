from flask import Flask, Response, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import load_model
from tempfile import NamedTemporaryFile
import os
import base64

app = Flask(__name__)

# Carregar o modelo VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Carregar o modelo treinado
CNN_Model = load_model(r'C:\Users\pytho\Documents\GitHub\Icity\iVision\Model\Model')

# Funções de preparação e previsão para a detecção de acidentes em tempo real
def prepared_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = base_model.predict(frame)
    return features.flatten().reshape(1, -1)

def predict_accident(frame):
    frame_ready = prepared_frame(frame)
    prediction = CNN_Model.predict(frame_ready)
    return prediction

# Funções para streaming de vídeo em tempo real
camera = cv2.VideoCapture(0)

def detect_accident(video_path, CNN_Model):
    cap = cv2.VideoCapture(video_path)
    accident_flag = False
    frame_window = []
    frame_skip = 5  # alterado para 5
    current_frame = 0
    max_prob = 0
    accident_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            if current_frame % frame_skip == 0:
                frame_window.append(frame)
            if len(frame_window) == 10:
                avg_frame = np.mean(frame_window, axis=0).astype(np.uint8)
                prediction = predict_accident(avg_frame)
                max_prob = max(max_prob, prediction[0][0])
                text = f'Probabilidade de Acidente: {max_prob*100:.2f}%'
                color = (0, 0, 255) if max_prob > 0.5 else (0, 255, 0)
                cv2.putText(avg_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                if prediction[0][0] > 0.5:
                    accident_flag = True
                    accident_frame = avg_frame
                    cv2.imshow('Acidente Detectado', avg_frame)
                    cv2.waitKey(0)
                    break
                else:
                    cv2.imshow('Sem Acidente', avg_frame)
                    cv2.waitKey(1)
                frame_window.pop(0)
    except Exception as e:
        print(f"Erro ao processar o vídeo: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return max_prob, accident_flag, accident_frame







def gen_frames(camera):  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            prediction = predict_accident(frame)
            max_prob = prediction[0][0]
            text = f'Probabilidade de Acidente: {max_prob*100:.2f}%'
            color = (0, 0, 255) if max_prob > 0.5 else (0, 255, 0)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/hello', methods=['GET'])
def HelloWorld():
    return 'Hello World'

@app.route('/', methods=['GET', 'POST'])
def home():
    return send_from_directory('template', 'MVP.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload', methods=['POST'])
def upload():
    video = request.files['video']
    temp_video_file = NamedTemporaryFile(delete=False)
    video.save(temp_video_file.name)
    max_prob, accident_flag, accident_frame = detect_accident(video_path=temp_video_file.name, CNN_Model=CNN_Model)
    _, buffer = cv2.imencode('.jpg', accident_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    if accident_flag:
        return jsonify({'status': 'danger', 'message': f"Acidente detectado com probabilidade de {100 * max_prob:.2f}%!", 'pred': float(max_prob), 'frame': jpg_as_text})
    else:
        return jsonify({'status': 'safe', 'message': f"Sem acidentes detectados. Probabilidade de segurança: {100 * (1 - max_prob):.2f}%", 'pred': 1 - float(max_prob), 'frame': jpg_as_text})


if __name__ == '__main__':
    app.run(debug=True)
