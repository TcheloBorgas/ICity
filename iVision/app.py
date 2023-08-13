#━━━━━━━━━❮Bibliotecas❯━━━━━━━━━
import os
import cv2
import numpy as np
import tensorflow.keras.backend as K
import time
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from twilio.rest import Client
from tempfile import NamedTemporaryFile
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from dotenv import load_dotenv
#━━━━━━━━━━━━━━❮◆❯━━━━━━━━━━━━━━

app = Flask(__name__)
load_dotenv(r'Code\amb_var.env')

CNN_Model = load_model(r'iVision\Model\Model')
account_sid = 'TACa583032405cbf44ef280fccae8db749e'
auth_token = 'a9b7afb38f6602c49252deba3cee4d5a'
client = Client(account_sid, auth_token)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

#━━━━━━❮Funções Principais❯━━━━━━━
def prepared_frame_v2(frame, base_model):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    features = base_model.predict(frame)
    return features.flatten().reshape(1, -1)

def detect_accident(video_path, CNN_Model, client):
    cap = cv2.VideoCapture(video_path)
    accident_flag = False
    frame_window = []
    frame_skip = 2  # Processar cada segundo quadro
    current_frame = 0

    try:
        while True:
            ret, frame = cap.read()
            current_frame += 1
            if current_frame % frame_skip != 0:
                continue
            if ret == True:
                frame_ready = prepared_frame_v2(frame, base_model)
                prediction = CNN_Model.predict(frame_ready)
                frame_window.append(prediction[0][0])
                if len(frame_window) > 15:  # Aumentando o tamanho da janela para 15
                    frame_window.pop(0)
                
                avg_prediction = np.mean(frame_window)

                if avg_prediction > 0.7:  # 35% de probabilidade
                    accident_flag = True

                if accident_flag:
                    predict = f"Probabilidade de acidente: {100 * avg_prediction:.2f}%"
                else:
                    predict = "Sem acidente"

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
    
    return avg_prediction, accident_flag

#━━━━━━❮ROTA TESTE❯━━━━━━━
@app.route('/hello', methods=['GET'])
def HelloWorld():
    return 'Hello World'

#━━━━━━❮ROTA Render❯━━━━━━━
@app.route('/', methods=['GET', 'POST'])
def home():
    return send_from_directory('template', 'MVP.html')

#━━━━━━❮ROTA Upload❯━━━━━━━
@app.route('/api/upload', methods=['POST'])
def upload():
    video = request.files['video']
    temp_video_file = NamedTemporaryFile(delete=False)
    video.save(temp_video_file.name)
    prob, accident_flag = detect_accident(video_path=temp_video_file.name, CNN_Model=CNN_Model, client=client)
    if accident_flag:
        return jsonify({'status': 'danger', 'message': 'Acidente detectado!', 'pred': float(prob)})
    else:
        return jsonify({'status': 'safe', 'message': 'Sem acidentes detectados.', 'pred': 0.0})

if __name__ == '__main__':
    app.run(debug=True)
