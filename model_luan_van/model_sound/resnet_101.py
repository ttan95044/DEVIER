import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.src.applications import  ResNet101
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow.keras as keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.models import Model
from keras.utils import to_categorical
import tensorflow.keras as keras
from keras.applications.inception_v3 import InceptionV3 , preprocess_input
from keras.layers import Dense , UpSampling2D , BatchNormalization , Dropout , Flatten
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
import time


# Đường dẫn đến thư mục chứa dữ liệu ảnh
path = "D:/luan-van-data/sound_img/"
# Danh sách các thư mục lớp
categories = ["cham", "hoa", "khmer", "kinh", "khac"]

class_names = categories

data = []  # Dữ liệu ảnh
labels = []  # Nhãn của ảnh

# Kích thước ảnh đầu vào
HEIGHT = 64
WIDTH = 64
CHANNELS = 3

# Duyệt qua các thư mục lớp và tạo danh sách ảnh và nhãn
for k, category in enumerate(categories):
    for f in os.listdir(path + category):
        image_path = os.path.join(path, category, f)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (WIDTH, HEIGHT))
                data.append(image)
                labels.append(k)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Hiển thị ảnh mẫu
plt.subplots(3, 4)
for i in range(12):
    plt.subplot(3, 4, i + 1)
    # Chuyển đổi độ sâu của ảnh từ CV_64F sang CV_8U
    plt.imshow(cv2.cvtColor(np.uint8(data[i] * 255), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(categories[labels[i]])

# plt.show()
# Tạo và biên dịch mô hình
base_model = ResNet101(include_top=False, weights='imagenet', input_shape=(HEIGHT, WIDTH, CHANNELS))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Đóng băng các layer của mô hình cơ sở
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

results1 = []  # Danh sách lưu trữ accuracy
results2 = []  # Danh sách lưu trữ recall
results3 = []  # Danh sách lưu trữ precision
results4 = []  # Danh sách lưu trữ F1 score

for e in range(1, 2):  # Lặp qua 10 lần huấn luyện
    start_time = time.time()

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

    # Chuyển đổi nhãn thành one-hot encoding
    trainY = to_categorical(trainY, len(class_names))
    testY = to_categorical(testY, len(class_names))

    # Huấn luyện mô hình
    history = model.fit(trainX, trainY,validation_split=0.2, batch_size=64, epochs=100, verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])
    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()

    # Đánh giá mô hình trên tập kiểm tra
    test_loss, test_accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"Lần {e}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    # Đánh giá và lưu trữ các chỉ số đánh giá
    predictions = model.predict(testX)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(testY, axis=1)

    end_time_test = time.time()  # Kết thúc đo thời gian kiểm tra
    training_time = end_time - start_time
    testing_time = end_time_test - start_time_test
    print("Thời gian huấn luyện:", training_time, "giây")
    print("Thời gian kiểm tra:", testing_time, "giây")

    accuracy = accuracy_score(true_labels, predicted_labels)
    results1.append(accuracy)

    recall = recall_score(true_labels, predicted_labels, average='weighted')
    results2.append(recall)

    precision = precision_score(true_labels, predicted_labels, average='weighted')
    results3.append(precision)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    results4.append(f1)

# Tính trung bình của 10 lần huấn luyện
average_accuracy = sum(results1) / len(results1)
average_recall = sum(results2) / len(results2)
average_precision = sum(results3) / len(results3)
average_f1 = sum(results4) / len(results4)

# In kết quả trung bình
print("Trung bình của 5 lần huấn luyện accuracy:", average_accuracy * 100)
print("Trung bình của 5 lần huấn luyện recall:", average_recall * 100)
print("Trung bình của 5 lần huấn luyện precision:", average_precision * 100)
print("Trung bình của 5 lần huấn luyện F1:", average_f1 * 100)
