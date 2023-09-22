import os
import cv2
import numpy as np
import tensorflow.keras.backend as K
import time
import threading
import atexit
import boto3
from flask import Flask, request, jsonify, send_from_directory, Response
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from moviepy.editor import ImageSequenceClip
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv('amb_var.env')
video_buffer = []
FPS = 30  # Assumindo que a câmera tenha 30 FPS, ajustar se necessário
BUFFER_SIZE = 15 * FPS  # 10 segundos de vídeo


CNN_Model = load_model('iVision\Model\Model')
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))



kinesis_client = boto3.client('kinesis', region_name='YOUR_REGION', 
                              aws_access_key_id='YOUR_ACCESS_KEY',
                              aws_secret_access_key='YOUR_SECRET_KEY')
stream_name = 'YOUR_STREAM_NAME'



if not os.path.exists("iVision\\temp"):
    os.makedirs("iVision\\temp")

temp_folder = 'iVision\\temp'


class Camera:
    accident_detected =False
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.frame = None
        self.accident_flag = False
        self.max_prob = 0
        self.lock = threading.Lock()


    def run(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break
            else:
                prediction = predict_accident(frame)
                max_prob = prediction[0][0]
                with self.lock:
                    self.frame = frame
                    self.max_prob = max_prob
                    self.accident_flag = max_prob > 0.3       # limiar de decisão
                    Camera.accident_detected = self.accident_flag
            if self.accident_flag:
                clip_name = "accident_clip.avi"
                clip_path = os.path.join(temp_folder, clip_name)
                save_buffer_as_video(video_buffer, clip_path)

camera = Camera()

def fetch_frame_from_kinesis():
    response = kinesis_client.get_records(
        ShardIterator='YOUR_SHARD_ITERATOR',
        Limit=1
    )
    
    # Aqui, assumimos que o frame é o único registro no lote e está codificado em base64.
    # Se sua configuração for diferente, ajuste conforme necessário.
    frame_data = response['Records'][0]['Data']
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    return frame

sns_client = boto3.client('sns', region_name='YOUR_REGION',
                          aws_access_key_id='YOUR_ACCESS_KEY',
                          aws_secret_access_key='YOUR_SECRET_KEY')
topic_arn = 'YOUR_SNS_TOPIC_ARN'

def send_accident_notification():
    message = "Um acidente foi detectado!"
    response = sns_client.publish(
        TopicArn=topic_arn,
        Message=message,
    )


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

                if avg_prediction > 0.3:  # limiar de decisão
                    accident_flag = True

                if accident_flag:
                    predict = f"Probabilidade de acidente: {100 * avg_prediction:.2f}%"
                    send_accident_notification()
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

def gen_frames(camera):
    global video_buffer
    clip_saved = False  # Indicador para evitar salvar clipes repetidos do mesmo evento
    while True:
        with camera.lock:
            frame = camera.frame
            # Adicionando o frame ao buffer
            video_buffer.append(frame)
            if len(video_buffer) > 10 * FPS:
                video_buffer.pop(0)
            max_prob = camera.max_prob
            accident_flag = camera.accident_flag

        if frame is None:
            continue

        text = f'Probabilidade de Acidente: {max_prob*100:.2f}%'
        color = (0, 0, 255) if accident_flag else (0, 255, 0)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Salvar um clipe se um acidente for detectado
        if accident_flag and not clip_saved:
            clip_name = "accident_clip.avi"
            clip_path = os.path.join(temp_folder, clip_name)
            save_buffer_as_video(video_buffer[-10 * FPS:], clip_path)
            clip_saved = True
        elif not accident_flag:
            clip_saved = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def save_buffer_as_video(buffer, filename):
    if not buffer:
        return

    # Verificar se todos os quadros são válidos
    if any(frame is None for frame in buffer):
        print("Erro: buffer de vídeo contém quadros inválidos.")
        return

    # Se todos os quadros forem válidos, continue com a criação do clipe
    fps = FPS
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in buffer], fps=fps)
    clip.write_videofile(filename, codec='libx264')




@app.route('/hello', methods=['GET'])
def HelloWorld():
    return 'Hello World'

@app.route('/', methods=['GET', 'POST'])
def home():
    return send_from_directory('template', 'Final_MVP.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    video = request.files['video']
    temp_video_file = NamedTemporaryFile(delete=False)
    video.save(temp_video_file.name)
    max_prob, accident_flag = detect_accident(video_path=temp_video_file.name, CNN_Model=CNN_Model)
    if accident_flag:
        send_accident_notification()
        return jsonify({'status': 'danger', 'message': f"Acidente detectado com probabilidade de {100 * max_prob:.2f}%!", 'pred': float(max_prob)})

    else:
        return jsonify({'status': 'safe', 'message': f"Sem acidentes detectados. Probabilidade de segurança: {100 * (1 - max_prob):.2f}%", 'pred': 1 - float(max_prob)})


@app.route('/video_feed')
def video_feed():
    def generate(camera):
        global video_buffer
        clip_saved = False  # Indicador para evitar salvar clipes repetidos do mesmo evento
        accident_detected_time = None

        while True:
            with camera.lock:
                frame = camera.frame
                # Adicionando o frame ao buffer
                video_buffer.append(frame)
                if len(video_buffer) > 10 * FPS:  # Mantendo o buffer com 10 segundos de quadros
                    video_buffer.pop(0)

                max_prob = camera.max_prob
                accident_flag = camera.accident_flag

            if frame is None:
                continue

            text = f'Probabilidade de Acidente: {max_prob*100:.2f}%'
            color = (0, 0, 255) if accident_flag else (0, 255, 0)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Salvar um clipe se um acidente for detectado
            if accident_flag:
                if not accident_detected_time:
                    accident_detected_time = time.time()
                elif time.time() - accident_detected_time > 5:  # 5 segundos após a detecção
                    clip_name = "accident_clip.avi"
                    clip_path = os.path.join(temp_folder, clip_name)
                    save_buffer_as_video(video_buffer, clip_path)
                    clip_saved = True
                    accident_detected_time = None
            else:
                clip_saved = False
                accident_detected_time = None

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        pass

    return Response(generate(Camera()), mimetype="multipart/x-mixed-replace")



@app.route('/accident_status')
def accident_status():
    return jsonify({
        "accident_detected": bool(Camera.accident_detected)  # Convertendo para booleano padrão
    })

@app.route('/accident_clip')
def accident_clip():
    return send_from_directory('tmp', "accident_clip.avi", as_attachment=False)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)