from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import time
import threading

app = Flask(__name__)

# Configurações
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Criar pastas se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Carregar modelo YOLO
model = YOLO('C:/Users/CASA/PycharmProjects/PythonProject/runs/detect/treinamento_customizado2/weights/best.pt')

# Variável para controlar o estado da detecção em tempo real
real_time_active = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}


def real_time_detection():
    global real_time_active
    cap = cv2.VideoCapture(0)  # 0 para webcam padrão

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        real_time_active = False
        return

    cv2.namedWindow('YOLO Real-time Detection', cv2.WINDOW_NORMAL)

    try:
        while real_time_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar detecção
            results = model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Mostrar resultado
            cv2.imshow('YOLO Real-time Detection', annotated_frame)

            # Sair com 'q' ou se a janela for fechada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('YOLO Real-time Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        real_time_active = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/realtime/start', methods=['POST'])
def start_real_time():
    global real_time_active
    if not real_time_active:
        real_time_active = True
        threading.Thread(target=real_time_detection, daemon=True).start()
        return jsonify({'success': True, 'message': 'Real-time detection started'})
    return jsonify({'success': False, 'message': 'Real-time detection is already running'})


@app.route('/api/realtime/stop', methods=['POST'])
def stop_real_time():
    global real_time_active
    real_time_active = False
    return jsonify({'success': True, 'message': 'Real-time detection stopped'})


@app.route('/api/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Tipo de arquivo não suportado'}), 400

    try:
        # Gerar nomes únicos para os arquivos
        file_id = str(uuid.uuid4())
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        original_filename = f"{file_id}_original.{original_ext}"
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)

        # Salvar arquivo original
        file.save(original_path)

        # Determinar se é vídeo
        video_flag = is_video(file.filename)

        if video_flag:
            # Processamento de vídeo
            cap = cv2.VideoCapture(original_path)
            if not cap.isOpened():
                raise Exception("Não foi possível abrir o vídeo original")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Forçar extensão .mp4 para o resultado
            result_filename = f"{file_id}_result.mp4"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)

            # Codec mais compatível (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

            if not out.isOpened():
                cap.release()
                raise Exception("Não foi possível criar o vídeo de saída")

            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=0.5)
                    annotated_frame = results[0].plot()

                    if annotated_frame is None:
                        raise Exception("Erro ao processar frame do vídeo")

                    out.write(annotated_frame)
                    processed_frames += 1

            finally:
                cap.release()
                out.release()

        else:
            # Processamento de imagem
            result_filename = f"{file_id}_result.{original_ext}"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            results = model.predict(original_path, conf=0.5)
            for r in results:
                im_array = r.plot()
                cv2.imwrite(result_path, im_array)

        return jsonify({
            'success': True,
            'original': f'/static/uploads/{original_filename}',
            'result': f'/static/results/{result_filename}',
            'is_video': video_flag
        })

    except Exception as e:
        # Limpar arquivos em caso de erro
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'result_path' in locals() and os.path.exists(result_path):
            os.remove(result_path)

        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


# Adicione estas novas rotas (mantendo todas as existentes)

@app.route('/api/videoprocess/start', methods=['POST'])
def start_video_processing():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    if not (file and allowed_file(file.filename) and is_video(file.filename)):
        return jsonify({'error': 'File type not supported'}), 400

    try:
        # Salvar vídeo temporariamente
        temp_id = str(uuid.uuid4())
        temp_filename = f"{temp_id}_temp.mp4"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)

        # Iniciar processamento em uma thread separada
        threading.Thread(
            target=process_video_realtime,
            args=(temp_path,),
            daemon=True
        ).start()

        return jsonify({
            'success': True,
            'message': 'Video processing started',
            'video_id': temp_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_video_realtime(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    cv2.namedWindow('Video Processing - YOLO', cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Processar frame com YOLO
            results = model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Mostrar resultado
            cv2.imshow('Video Processing - YOLO', annotated_frame)

            # Sair com 'q' ou se a janela for fechada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Video Processing - YOLO', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Remover arquivo temporário
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == '__main__':
    app.run(debug=True)