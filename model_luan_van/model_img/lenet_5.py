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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras import optimizers
import tensorflow as tf
import time


path = "D:/luan-van-data/img/" #đường dẫm
categories_dantoc = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục
categories = ["cuoi", "mua", "chua", "cho_noi", "le"] #thư mục


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

# Duyệt qua danh mục và tạo danh sách đường dẫn
for k, category in enumerate(categories):
    for h in categories_dantoc:
        for f in os.listdir(path+category+"/"+h):
            # print("F=",f)
            # print("H=",h)
            imagePaths.append([path+category+'/'+h+"/"+f, k])
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
plt.subplots(3,4)

for i in range(12):#lặp qua 12 ảnh đầu tiên trong danh sách data
    plt.subplot(3,4, i+1)#tạo một ô con trong lưới 3x4 và chọn ô con thứ i+1 để hiển thị hình ảnh tiếp theo
    plt.imshow(data[i])#hiển thị hình ảnh thứ i từ danh sách data
    plt.axis('off')#tắt các dấu trục x và y để làm cho hình ảnh trở nên rõ ràng hơn
    plt.title(categories[labels[i]])#đặt tiêu đề cho hình ảnh dựa trên nhãn tương ứng từ danh sách categories và labels

# Hiển thị biểu đồ
# plt.show()

EPOCHS = 100
INIT_LR = 1e-3  # tốc độ học ban đầu
BS = 30  # kích thước bath

# defining the LENET-5 architecture
model = keras.Sequential()
model.add(Conv2D(6 , 5 , activation='tanh' , padding='same'))
model.add(MaxPool2D(2))
model.add(Conv2D(16 , 5 , activation = 'tanh' , padding='same'))
model.add(MaxPool2D(2))
model.add(Conv2D(120 , 5 , activation = 'tanh' , padding='same'))
model.add(Flatten())
model.add(Dense(84 , activation='tanh'))
model.add(Dense(len(class_names), activation='softmax'))


#initializing the optimzer
optim = optimizers.Adam()

# defining the loss and evalution metrics
model.compile(optim , loss=tf.keras.losses.categorical_crossentropy , metrics=['accuracy'])


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
print("Trung bình của 5 lần huấn luyện accuracy:", average_accuracy)

average_recall = sum(results2) / len(results2)
print("Trung bình của 5 lần huấn luyện recall:", average_recall)

average_precision = sum(results3) / len(results3)
print("Trung bình của 5 lần huấn luyện precision:", average_precision)

average_f1 = sum(results4) / len(results4)
print("Trung bình của 5 lần huấn luyện f1:", average_f1)
