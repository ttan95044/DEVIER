import numpy as np
import tensorflow as tf
from PIL import Image

# Đường dẫn đến tệp .tflite
model_path = 'svm_sound.tflite'

# Load model
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Kiểm tra đầu vào và đầu ra của model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kiểm tra kích thước đầu vào
input_shape = input_details[0]['shape']
print(f"Input shape: {input_shape}")

# Đọc và tiền xử lý hình ảnh đầu vào
image_path = 'static/random_10s.png'
image = Image.open(image_path)
image = image.resize((input_shape[1], input_shape[2]))
image = np.array(image, dtype=np.float32) / 255.0  # Chuyển đổi kiểu dữ liệu và chuẩn hóa hình ảnh

# Cắt bớt kênh alpha (kênh thứ tư)
image = image[:, :, :3]

# Kiểm tra hình ảnh
print(f"Input image shape: {image.shape}")

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

# In ra danh sách các nhãn và xác suất tương ứng
for label, probability in sorted_labels_with_probabilities:
    print(f"Label: {label}, Probability: {probability}")
