import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import tensorflow as tf
from keras.models import Sequential, Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
path = "D:/luan-van-data/img/mua/" #đường dẫm
# categories = ["cham", "hoa", "khmer"] #thư mục
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục

data = []#dữ liệu
labels = []#nhãn
imagePaths = []#danh sách ảnh có đường dẫn

#định dạng kích thước ảnh
HEIGHT = 128
WIDTH = 128
# 24 24
N_CHANNELS = 3#màu RGB
# Duyệt qua danh mục và tạo danh sách đường dẫn
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

#random ảnh
import random
random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    if not os.path.isfile(imagePath[0]):
        print("Error: File not found -", imagePath[0])
        continue
    image = cv2.imread(imagePath[0])
    if image is None:
        print("Error: Unable to read image -", imagePath[0])
        continue

#đọc và xử lý hình ảnh từ danh sách imagePaths
for imagePath in imagePaths:#duyệt qua từng imagepath
    image = cv2.imread(imagePath[0])#đọc ảnh từ đường dẫn
    if image is None:
        print("Error: Unable to read image -", imagePath[0])
        continue
    if image.size == 0:  # Kiểm tra kích thước ảnh
        print("Error: Empty image -", imagePath[0])
        continue
    image = cv2.resize(image, (WIDTH, HEIGHT))  # thay đổi kích thước ảnh
    data.append(image)# thêm vào danh sách data
    label = imagePath[1] #gán nhãn cho dữ liệu tương ứng
    labels.append(label)
# chuyển đổi danh sách data và labels thành các mảng NumPy. Hình ảnh trong data được chuẩn hóa bằng cách chia cho 255 để nằm trong khoảng từ 0 đến 1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)



EPOCHS = 100
INIT_LR = 1e-3
BS =14

class_names = categories


model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))
# print(model.summary())

opt=tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

results1 = []
results2 = []
results3 = []
results4 = []


for e in range(1,2):
    start_time = time.time()
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=10, shuffle=True)

    # Chuyển đổi nhãn thành one-hot encoding
    trainY_one_hot = to_categorical(trainY, num_classes=len(class_names))
    testY_one_hot = to_categorical(testY, num_classes=len(class_names))

    # Định nghĩa early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Sử dụng trainY_one_hot và testY_one_hot trong model.fit
    # model.fit(trainX, trainY_one_hot, batch_size=BS, epochs=EPOCHS, verbose=1)
    history = model.fit(trainX, trainY_one_hot, validation_split=0.1, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()  # Bắt đầu đo thời gian kiểm tra
    # print("bat dau tric dat trung")
    new_model=Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    # print("new_model: ", new_model)

    feat_train=new_model.predict(trainX)
    # print("feat_train",feat_train.shape)

    feat_test=new_model.predict(testX)
    # print("feat_test",feat_test.shape)

    # print (" khoi tao SVR")
    # model_SVM=SVC(kernel="rbf", C=10000, gamma=0.001)
    # model_SVM.fit(feat_train,np.argmax(trainY,axis=1))
    # prepY=model_SVM.predict(feat_test)

    # print("Khoi tao SVM")
    model_SVM = SVC(kernel="rbf", C=100000, gamma=0.01)
    # Chuyển đổi one-hot encoding về mảng một chiều
    trainY_single = np.argmax(trainY_one_hot, axis=1)
    testY_single = np.argmax(testY_one_hot, axis=1)

    # Sử dụng trainY_single trong model_SVM.fit
    model_SVM.fit(feat_train, trainY_single)
    prepY = model_SVM.predict(feat_test)
    # print("day ket qua dat trung",prepY)
    end_time_test = time.time()  # Kết thúc đo thời gian kiểm tra
    training_time = end_time - start_time
    testing_time = end_time_test - start_time_test
    print("Thời gian huấn luyện:", training_time, "giây")
    print("Thời gian kiểm tra:", testing_time, "giây")
    accuracy = accuracy_score(testY, prepY)
    results1.append(accuracy)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    recall = recall_score(testY, prepY, average='weighted')
    results2.append(recall)
    # print("Recall: %.2f%%" % (recall * 100.0))

    precision = precision_score(testY, prepY, average='weighted')
    results3.append(precision)
    # print("Precision: %.2f%%" % (precision * 100.0))

    f1 = f1_score(testY, prepY, average='weighted')
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
