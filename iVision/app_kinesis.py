import os
import cv2
import numpy as np
import tensorflow.keras.backend as K
from flask import Flask, request, jsonify, send_from_directory, Response
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import boto3
import time
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv('amb_var.env')


CNN_Model = load_model('iVision/Model/Model')
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

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

def kinesis_video_feed():
    client = boto3.client('kinesisvideo', region_name='us-west-2')  # Use sua região
    response = client.get_data_endpoint(
        StreamARN='YOUR_STREAM_ARN',  # Use o ARN do seu stream
        APIName='GET_MEDIA'
    )

    endpoint = response['DataEndpoint']
    client = boto3.client('kinesis-video-media', endpoint_url=endpoint, region_name='us-west-2')  # Use sua região
    response = client.get_media(
        StreamARN='YOUR_STREAM_ARN',  # Use o ARN do seu stream
        StartSelector={
            'StartSelectorType': 'NOW'
        }
    )

    for chunk in response['Payload'].iter_chunks():
        frame = np.frombuffer(chunk, np.uint8)
        prediction = predict_accident(frame)
        max_prob = prediction[0][0]
        accident_flag = max_prob > 0.02

        text = f'Probabilidade de Acidente: {max_prob*100:.2f}%'
        color = (0, 0, 255) if accident_flag else (0, 255, 0)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_accident(video_path, CNN_Model):
    cap = cv2.VideoCapture(video_path)
    accident_flag = False
    max_prediction = 0
    frame_window = []
    frame_skip = 2  # Processar cada segundo quadro
    current_frame = 0

    try:
        while True:
            ret, frame = cap.read()
            current_frame += 1
            if current_frame % frame_skip != 0:
                continue
            if ret:
                frame_ready = prepared_frame(frame)
                prediction = CNN_Model.predict(frame_ready)
                frame_window.append(prediction[0][0])
                if len(frame_window) > 15:
                    frame_window.pop(0)
                
                max_prediction = max(max_prediction, prediction[0][0])
                avg_prediction = np.mean(frame_window)

                if avg_prediction > 0.01:  # limiar de decisão
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
            time.sleep(1.25)
            os.unlink(video_path)
        except PermissionError:
            print("Não foi possível excluir o arquivo. Ele pode estar sendo usado por outro processo.")
    
    return max_prediction, accident_flag

@app.route('/video_feed')
def video_feed():
    return Response(kinesis_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload', methods=['POST'])
def upload():
    video = request.files['video']
    temp_video_file = NamedTemporaryFile(delete=False)
    video.save(temp_video_file.name)
    max_prob, accident_flag = detect_accident(video_path=temp_video_file.name, CNN_Model=CNN_Model)
    if accident_flag:
        return jsonify({'status': 'danger', 'message': f"Acidente detectado com probabilidade de {100 * max_prob:.2f}%!", 'pred': float(max_prob)})
    else:
        return jsonify({'status': 'safe', 'message': f"Sem acidentes detectados. Probabilidade de segurança: {100 * (1 - max_prob):.2f}%", 'pred': 1 - float(max_prob)})

@app.route('/')
def home():
    return send_from_directory('template', 'MVP.html')

if __name__ == '__main__':
    app.run(debug=True)
