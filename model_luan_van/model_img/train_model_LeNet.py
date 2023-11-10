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
import random
from keras.callbacks import EarlyStopping
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, DenseNet121, EfficientNetB7
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from keras import layers
# from keras import models
import time


path = "D:/luan-van-data/img/" #đường dẫm
categories_dantoc = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục
categories = ["cuoi", "mua", "chua", "cho_noi", "le"] #thư mục

#tien xu ly du lieu va nhan dau vao
data = []#dữ liệu
labels = []#nhãn
imagePaths = []#danh sách ảnh có đường dẫn

#định dạng kích thước ảnh
HEIGHT = 128
WIDTH = 128

N_CHANNELS = 1 # 1 màu
# Duyệt qua danh mục và tạo danh sách đường dẫn
for k, category in enumerate(categories):
    for h in categories_dantoc:
        for f in os.listdir(path+category+"/"+h):
            # print("F=",f)
            # print("H=",h)
            imagePaths.append([path+category+'/'+h+"/"+f, k])

#random ảnh
random.shuffle(imagePaths)
print(imagePaths[:10])
# print("Chuan bi doc anh tu folder: ")

#đọc và xử lý hình ảnh từ danh sách imagePaths
for imagePath in imagePaths:#duyệt qua từng imagepath
      image = cv2.imread(imagePath[0])#đọc ảnh từ đường dẫn
      image = cv2.resize(image, (WIDTH, HEIGHT))# thay đổi kích thước ảnh
      image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# chuyển đổi ảnh màu thành ảnh xám
      data.append(image_gray)# thêm vào danh sách data
      label = imagePath[1]#gán nhãn cho dữ liệu tương ứng
      labels.append(label)


# chuyển đổi danh sách data và labels thành các mảng NumPy. Hình ảnh trong data được chuẩn hóa bằng cách chia cho 255 để nằm trong khoảng từ 0 đến 1
# print("scale raw pixel / 255.0")
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

# print("train test split")


#print("TrainY: {}".format(trainY))
# print("trainX: ", trainX.shape)
# print("testX: ", testX.shape)
# print("trainY: ", trainY.shape)
# print("testY: ", testY.shape)
# print("valX: ", valX.shape)
# print("valY: ", valY.shape)


EPOCHS = 100
INIT_LR = 1e-3
BS = 30


# print("[INFO] compiling model...")

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, N_CHANNELS)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile mô hình
opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

results1 = []
results2 = []
results3 = []
results4 = []

for e in range(1,2):
    start_time = time.time()

    # Chia dữ liệu thành tập train và test
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.1, random_state=10)
    # Chuyển đổi các nhãn lớp thành dạng one-hot encoding
    trainY = to_categorical(trainY, len(categories))
    testY = to_categorical(testY, len(categories))
    # print("to_categorical  TrainY: {}".format(trainY))
    # Chia tập train thành tập train và validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=10)
    # Định nghĩa early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    #huan luyen mo hinh
    # print("bat dau huan luyen")

    history = model.fit(trainX, trainY, validation_split=0.1, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()  # Bắt đầu đo thời gian kiểm tra
    # model.save('./model_lenet_trained/cnn_lenet_ep{}_bs{}.h5'.format(EPOCHS, BS))

    # print("Ve do thi CNN:  ")
    # plot loss
    # plt.plot(history.history['loss'], color='red', linewidth=3, label='loss')
    # plt.plot(history.history['accuracy'], color='cyan', linewidth=3, label='accuracy')
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(loc='upper right')
    # # plt.savefig('./model_extract_feature_CNN/lenet_extract_features_valLossMAE_epo{}_bs{}.png'.format(EPOCHS, BS), dpi=300)
    # plt.show()

    # import os
    #
    # save_dir = './fig_Accuracy_Lenet/'
    # os.makedirs(save_dir, exist_ok=True)

    #danh gia mo hinh
    # print("Bat dau du doan mo hinh")

    pred = model.predict(testX)#dự đoán đầu ra cho dữ liệu thử nghiệm (testX) bằng cách sử dụng mô hình máy học
    predictions = argmax(pred, axis=1)#xác định lớp dự đoán cho từng mẫu dữ liệu thử nghiệm
    cm = confusion_matrix(np.argmax(testY, axis=1), predictions)#tạo ra ma trận nhầm lẫn (confusion matrix) bằng cách so sánh dự đoán (predictions) với nhãn thực tế (testY) của dữ liệu thử nghiệm
    # fig = plt.figure()#tạo một đối tượng hình ảnh
    # ax = fig.add_subplot(111)#tạo một đối tượng subplot (phần con) trong hình ảnh với chỉ số 111
    # cax = ax.matshow(cm)#Đoạn mã này vẽ ma trận nhầm lẫn lên đối tượng subplot ax bằng cách sử dụng hàm matshow
    # plt.title('Model confusion matrix')#Đoạn mã này đặt tiêu đề cho biểu đồ ma trận nhầm lẫn
    # fig.colorbar(cax)#thêm thanh màu (colorbar) vào biểu đồ để hiển thị giá trị tương ứng với màu sắc trên biểu đồ ma trận nhầm lẫn.
    # # Đặt nhãn cho trục x của biểu đồ, trong đó categories là danh sách các lớp hoặc nhãn trong bài toán phân loại. Nhãn này giúp bạn hiểu được ô tương ứng với mỗi lớp
    # plt.xticks(np.arange(len(categories)), categories)
    # #Đặt nhãn cho trục y của biểu đồ, cũng sử dụng danh sách categories để hiển thị nhãn cho từng dòng trong ma trận nhầm lẫn.
    # plt.yticks(np.arange(len(categories)), categories)

    #thêm các giá trị từ ma trận nhầm lẫn (cm) vào các ô của biểu đồ ma trận nhầm lẫn đã được tạo trước đó
    # for i in range(len(categories)):
    #   for j in range(len(categories)):
    #     ax.text(i, j, cm[j, i], va='center', ha='center')#tạo ra một đoạn văn bản và đặt nó tại vị trí (i, j) trên biểu đồ ma trận nhầm lẫn (
    #
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.savefig("D:/luan-van-model/model/fig_Accuracy_Lenet/confusion_matrix_epo{}_bs{}_bo_exception.png".format(EPOCHS, BS))
    # plt.show()

    end_time_test = time.time()  # Kết thúc đo thời gian kiểm tra
    training_time = end_time - start_time
    testing_time = end_time_test - start_time_test
    print("Thời gian huấn luyện:", training_time, "giây")
    print("Thời gian kiểm tra:", testing_time, "giây")

    # Assume y_true and y_pred are the true labels and predicted labels, respectively
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
