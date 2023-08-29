# Importações
from flask import Flask, Response, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import load_model
from twilio.rest import Client
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import tensorflow.keras.backend as K
import os
import threading

app = Flask(__name__)

# Carregar variáveis de ambiente
load_dotenv(r'Code\amb_var.env')

# Carregar o modelo VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Carregar o modelo treinado
CNN_Model = load_model('iVision\Model\Model')

# Configurações do Twilio
account_sid = 'TACa583032405cbf44ef280fccae8db749e'
auth_token = 'a9b7afb38f6602c49252deba3cee4d5a'
client = Client(account_sid, auth_token)

# Funções de preparação e previsão para a detecção de acidentes em tempo real
def prepared_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = base_model.predict(frame)
    return features.flatten().reshape(1, -1)

def prepared_frame_v2(frame, base_model):
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
global_frame = None

def capture_frames():
    global global_frame, camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            global_frame = frame

def process_frames():
    global global_frame
    while True:
        if global_frame is not None:
            prediction = predict_accident(global_frame)
            label = "Acidente" if prediction[0][0] > 0.02 else "Sem Acidente"
            global_frame = cv2.putText(global_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

@app.route('/video_feed')
def video_feed():
    thread1 = threading.Thread(target=capture_frames)
    thread2 = threading.Thread(target=process_frames)
    
    thread1.start()
    thread2.start()
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global global_frame
    while True:
        if global_frame is not None:
            _, buffer = cv2.imencode('.jpg', global_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_accident(video_path, CNN_Model, client):
    cap = cv2.VideoCapture(video_path)
    accident_flag = False
    frame_window = []
    frame_skip = 1  # Processar cada segundo quadro
    current_frame = 0
    max_prob = 0  # Armazenar a probabilidade máxima de acidente

    try:
        while True:
            ret, frame = cap.read()
            current_frame += 1
            if current_frame % frame_skip != 0:
                continue
            if ret == True:
                frame_ready = prepared_frame_v2(frame, base_model)
                prediction = CNN_Model.predict(frame_ready)
                max_prob = max(max_prob, prediction[0][0])  # Atualizar a probabilidade máxima
                frame_window.append(prediction[0][0])
                if len(frame_window) > 15:
                    frame_window.pop(0)
                
                avg_prediction = np.mean(frame_window)

                if avg_prediction > 0.01:  # Limiar de 1%
                    accident_flag = True

                if accident_flag:
                    predict = f"Probabilidade de acidente: {100 * max_prob:.2f}%"
                else:
                    predict = f"Sem acidente (Probabilidade: {100 * (1 - avg_prediction):.2f}%)"

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
                cv2.imshow('Frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        K.clear_session()
        try:
            time.sleep(1)
            os.unlink(video_path)
        except PermissionError:
            print("Não foi possível excluir o arquivo. Ele pode estar sendo usado por outro processo.")
    
    return max_prob, accident_flag

# Rotas
@app.route('/hello', methods=['GET'])
def HelloWorld():
    return 'Hello World'

@app.route('/', methods=['GET', 'POST'])
def home():
    return send_from_directory('template', 'MVP.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    video = request.files['video']
    temp_video_file = NamedTemporaryFile(delete=False)
    video.save(temp_video_file.name)
    max_prob, accident_flag = detect_accident(video_path=temp_video_file.name, CNN_Model=CNN_Model, client=client)
    if accident_flag:
        return jsonify({'status': 'danger', 'message': f"Acidente detectado com probabilidade de {100 * max_prob:.2f}%!", 'pred': float(max_prob)})
    else:
        return jsonify({'status': 'safe', 'message': f"Sem acidentes detectados. Probabilidade de segurança: {100 * (1 - max_prob):.2f}%", 'pred': 1 - float(max_prob)})

if __name__ == '__main__':
    app.run(debug=True)
