import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time

path = "D:/luan-van-data/img/mua/" #đường dẫm
# categories = ["cham", "hoa", "khmer"] #thư mục
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục

class_names = categories #lớp tên thư mục
print(class_names)

data = []#dữ liệu
labels = []#nhãn
imagePaths = [] #danh sách ảnh có đường dẫn

#định dạng kích thước ảnh
HEIGHT = 128
WIDTH = 128

#màu RGB
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])
#random ảnh
import random
random.shuffle(imagePaths)
print(imagePaths[:10])


#đọc và xử lý hình ảnh từ danh sách imagePaths
for imagePath in imagePaths:#duyệt qua từng imagepath
    image = cv2.imread(imagePath[0])#đọc ảnh từ đường dẫn
    image = cv2.resize(image, (WIDTH, HEIGHT))  # thay đổi kích thước ảnh
    data.append(image)# thêm vào danh sách data
    label = imagePath[1] #gán nhãn cho dữ liệu tương ứng
    labels.append(label)

# chuyển đổi danh sách data và labels thành các mảng NumPy. Hình ảnh trong data được chuẩn hóa bằng cách chia cho 255 để nằm trong khoảng từ 0 đến 1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Vẽ ảnh trên lưới 3x4
# plt.subplots(3,4)
#
# for i in range(12):#lặp qua 12 ảnh đầu tiên trong danh sách data
#     plt.subplot(3,4, i+1)#tạo một ô con trong lưới 3x4 và chọn ô con thứ i+1 để hiển thị hình ảnh tiếp theo
#     plt.imshow(data[i])#hiển thị hình ảnh thứ i từ danh sách data
#     plt.axis('off')#tắt các dấu trục x và y để làm cho hình ảnh trở nên rõ ràng hơn
#     plt.title(categories[labels[i]])#đặt tiêu đề cho hình ảnh dựa trên nhãn tương ứng từ danh sách categories và labels
#
# # Hiển thị biểu đồ
# plt.show()

EPOCHS = 100
INIT_LR = 1e-3  # tốc độ học ban đầu
BS = 14  # kích thước bath

# Xây dựng mô hình neural network
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

results1 = []
results2 = []
results3 = []
results4 = []

for e in range(1,2):
    start_time = time.time()
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=10, shuffle=True)

    # Chuyển đổi các nhãn lớp thành dạng one-hot encoding
    trainY = to_categorical(trainY, len(class_names))
    testY = to_categorical(testY, len(class_names))
    #
    # print(trainX.shape)
    # print(testX.shape)
    # print(trainY.shape)
    # print(testY.shape)

    # Định nghĩa early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # print(model.summary())

    history = model.fit(trainX, trainY, validation_split=0.1, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()  # Bắt đầu đo thời gian kiểm tra
    # đoạn này dùng để save mô hình
    # model.save("lenet.h5")

    # đoạn này dùng để kiểm tra mô hình

    from numpy import argmax

    pred = model.predict(testX)#Dòng này dùng để dự đoán các nhãn cho dữ liệu
    predictions = np.argmax(pred, axis=1) #Dòng này dùng để chuyển đổi các giá trị dự đoán (pred) thành các nhãn lớp dự đoán
    end_time_test = time.time()  # Kết thúc đo thời gian kiểm tra
    training_time = end_time - start_time
    testing_time = end_time_test - start_time_test
    print("Thời gian huấn luyện:", training_time, "giây")
    print("Thời gian kiểm tra:", testing_time, "giây")
    # # Tính ma trận nhầm lẫn
    # cm = confusion_matrix(np.argmax(testY, axis=1), predictions)
    #
    # fig = plt.figure()#Tạo một hình vẽ mới.
    # ax = fig.add_subplot(111)# Thêm một trục (axes) vào hình vẽ. Số 111 ở đây cho biết bạn đang tạo một trục duy nhất trong hình vẽ.
    # cax = ax.matshow(cm)#Vẽ ma trận nhầm lẫn (cm) trên trục vừa được tạo ra. Hàm matshow được sử dụng để hiển thị ma trận nhầm lẫn dưới dạng biểu đồ ma trận.
    # plt.title('Model confusion matrix')#Đặt tiêu đề cho biểu đồ
    # fig.colorbar(cax)#Thêm thanh màu vào biểu đồ. Thanh màu này sẽ liên kết với giá trị trong ma trận nhầm lẫn và hiển thị một biểu đồ màu tương ứng.
    # # Đặt nhãn cho trục x của biểu đồ, trong đó categories là danh sách các lớp hoặc nhãn trong bài toán phân loại. Nhãn này giúp bạn hiểu được ô tương ứng với mỗi lớp
    # plt.xticks(np.arange(len(categories)), categories)
    # #Đặt nhãn cho trục y của biểu đồ, cũng sử dụng danh sách categories để hiển thị nhãn cho từng dòng trong ma trận nhầm lẫn.
    # plt.yticks(np.arange(len(categories)), categories)

    # for i in range(5):
    #     for j in range(5):
    #         ax.text(i, j, cm[j, i], va='center', ha='center')#Trong mỗi vòng lặp, đoạn mã này tạo ra một đoạn văn bản (text) và đặt nó tại vị trí (i, j) trên biểu đồ heatmap

    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    # In ra số epoch mà mô hình đã được huấn luyện
    # print("Số epoch đã được huấn luyện:", len(history.history['loss']))

    accuracy = accuracy_score(np.argmax(testY, axis=1), predictions)
    # print("Accuracy : %.2f%%" % (accuracy * 100.0))
    results1.append(accuracy)

    recall = recall_score(np.argmax(testY, axis=1), predictions, average='weighted')
    results2.append(recall)
    # print("Recall: %.2f%%" % (recall * 100.0))

    precision = precision_score(np.argmax(testY, axis=1), predictions, average='weighted')
    results3.append(precision)
    # print("Precision: %.2f%%" % (precision * 100.0))

    f1 = f1_score(np.argmax(testY, axis=1), predictions, average='weighted')
    results4.append(f1)
    # print("F1 Score: %.2f%%" % (f1 * 100.0))


average_accuracy = sum(results1) / len(results1)
print("Trung bình của 10 lần huấn luyện accuracy:", average_accuracy)

average_recall = sum(results2) / len(results2)
print("Trung bình của 10 lần huấn luyện recall:", average_recall)

average_precision = sum(results3) / len(results3)
print("Trung bình của 10 lần huấn luyện precision:", average_precision)

average_f1 = sum(results4) / len(results4)
print("Trung bình của 10 lần huấn luyện f1:", average_f1)
