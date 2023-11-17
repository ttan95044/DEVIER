import cv2
from flask import Flask, request, jsonify
import librosa.display
import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

# Đường dẫn lưu trữ audio và hình ảnh spectrogram sau xử lý
AUDIO_UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Biến để lưu trữ kết quả của dự đoán
result = []

# Hàm để cắt audio thành nhiều phần 10 giây
def cut_audio(audio_path, output_directory, duration):
    data, samplerate = sf.read(audio_path)
    num_parts = int(len(data) / (samplerate * duration))

    for i in range(num_parts):
        start = i * samplerate * duration
        end = min((i + 1) * samplerate * duration, len(data))
        part = data[start:end]
        output_file = os.path.join(output_directory, f'part_{i + 1}.wav')
        sf.write(output_file, part, samplerate)

# Hàm để tạo hình ảnh spectrogram từ audio
def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

# Hàm để dự đoán và lưu trữ kết quả từ một phần audio
def predict_audio_segment(audio_file, model_path, image_file):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Kiểm tra đầu vào và đầu ra của model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Kiểm tra kích thước đầu vào
    input_shape = input_details[0]['shape']

    # Đọc và tiền xử lý hình ảnh đầu vào từ image_file bằng OpenCV
    image = cv2.imread(image_file)
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = image.astype(np.float32) / 255.0  # Chuyển đổi kiểu dữ liệu và chuẩn hóa hình ảnh

    # Cắt bớt kênh alpha (kênh thứ tư)
    image = image[:, :, :3]

    # print("Kích thước ảnh sau khi xử lý:", image.shape)
    # plt.imshow(image)
    # plt.show()

    # Chạy model để dự đoán
    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Danh sách các nhãn
    categories = ["cham", "hoa", "khmer", "kinh", "khac"]

    # Lấy danh sách các nhãn và xác suất tương ứng
    labels_with_probabilities = list(zip(categories, output_data[0]))

    # Sắp xếp danh sách theo xác suất giảm dần
    sorted_labels_with_probabilities = sorted(labels_with_probabilities, key=lambda x: x[1], reverse=True)

    # Chọn nhãn có xác suất cao nhất
    top_label = sorted_labels_with_probabilities[0][0]

    return top_label


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Không tìm thấy tệp audio!'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Chưa chọn tệp audio!'})

    # Xóa tất cả các tệp .png và .wav trong thư mục static và uploads
    for file in os.listdir(STATIC_FOLDER):
        if file.endswith('.png') or file.endswith('.wav'):
            file_path = os.path.join(STATIC_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Không thể xóa tệp {file_path}: {e}")

    audio_path = os.path.join(AUDIO_UPLOAD_FOLDER, 'audio.wav')
    audio_file.save(audio_path)

    # Cắt audio thành nhiều phần 10 giây
    cut_audio(audio_path, AUDIO_UPLOAD_FOLDER, 10)

    # Dự đoán và lưu trữ kết quả từng phần audio
    results = []
    dir = os.listdir(AUDIO_UPLOAD_FOLDER)
    for file in dir:
        if file.endswith('.wav'):
            segment_audio = os.path.join(AUDIO_UPLOAD_FOLDER, file)
            # Tạo spectrogram từ audio và lưu nó
            spectrogram_image = os.path.join(STATIC_FOLDER, f'spectrogram_{file}.png')
            create_spectrogram(segment_audio, spectrogram_image)
            result_segment = predict_audio_segment(segment_audio, 'svm_sound1.tflite', spectrogram_image)
            results.append(result_segment)

    global result
    result = results  # Cập nhật biến result với kết quả từ yêu cầu POST

    return jsonify({'result': 'Xử lý audio thành công'})

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global result

    if not result:
        return jsonify({'error': 'Chưa có kết quả dự đoán'})

    # Tạo một từ điển để đếm số lần xuất hiện của từng nhãn trong kết quả
    label_counts = {}
    for item in result:
        if item in label_counts:
            label_counts[item] += 1
        else:
            label_counts[item] = 1

    # Sắp xếp các nhãn theo số lần xuất hiện giảm dần
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    # Lấy ra nhãn đúng nhất, nhãn đúng thứ hai và nhãn đúng thứ ba (nếu có)
    top_label = sorted_labels[0][0]
    second_top_label = sorted_labels[1][0] if len(sorted_labels) > 1 else None
    third_top_label = sorted_labels[2][0] if len(sorted_labels) > 2 else None

    return jsonify({
        'top_label': top_label,
        'second_top_label': second_top_label,
        # 'third_top_label': third_top_label,
        # 'all_lebals':label_counts,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
