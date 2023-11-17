import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
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

path = "D:/luan-van-data/sound_img/" #đường dẫm
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục

data = []#dữ liệu
labels = []#nhãn
imagePaths = []#danh sách ảnh có đường dẫn

#định dạng kích thước ảnh
HEIGHT = 64
WIDTH = 64
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

plt.subplots(3,4)
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
plt.show()


EPOCHS = 100
INIT_LR = 1e-3
BS =64

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


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=45, shuffle=True)

# Chuyển đổi nhãn thành one-hot encoding
trainY_one_hot = to_categorical(trainY, num_classes=len(class_names))
testY_one_hot = to_categorical(testY, num_classes=len(class_names))

# Định nghĩa early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
# Đo thời gian trước khi bắt đầu huấn luyện
start_time_train = time.time()

# Sử dụng trainY_one_hot và testY_one_hot trong model.fit
history = model.fit(trainX, trainY_one_hot, validation_split=0.1, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
# print("bat dau tric dat trung")
# Đo thời gian khi kết thúc quá trình huấn luyện
end_time_train = time.time()
# In ra thời gian huấn luyện
training_time = end_time_train - start_time_train
print("Thời gian huấn luyện:", training_time, "giây")
new_model=Model(inputs=model.input, outputs=model.get_layer('dense').output)
print("new_model: ", new_model)

feat_train=new_model.predict(trainX)
print("feat_train",feat_train.shape)

feat_test=new_model.predict(testX)
print("feat_test",feat_test.shape)

# print("Khoi tao SVM")
model_SVM = SVC(kernel="rbf", C=100000, gamma=0.01)
# Chuyển đổi one-hot encoding về mảng một chiều
trainY_single = np.argmax(trainY_one_hot, axis=1)
testY_single = np.argmax(testY_one_hot, axis=1)

# Sử dụng trainY_single trong model_SVM.fit
model_SVM.fit(feat_train, trainY_single)
# Đo thời gian trước khi bắt đầu quá trình kiểm tra
start_time_test = time.time()

# Thực hiện quá trình kiểm tra bằng model_SVM.predict
prepY = model_SVM.predict(feat_test)

# Đo thời gian khi kết thúc quá trình kiểm tra
end_time_test = time.time()

# In ra thời gian kiểm tra
testing_time = end_time_test - start_time_test
print("Thời gian kiểm tra:", testing_time, "giây")

accuracy = accuracy_score(testY, prepY)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

recall = recall_score(testY, prepY, average='weighted')
print("Recall: %.2f%%" % (recall * 100.0))

precision = precision_score(testY, prepY, average='weighted')
print("Precision: %.2f%%" % (precision * 100.0))

f1 = f1_score(testY, prepY, average='weighted')
print("F1 Score: %.2f%%" % (f1 * 100.0))

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# # Xác định đường dẫn tệp mà bạn muốn lưu mô hình TFLite
# tflite_model_path = "svm_sound2.tflite"
#
# # Lưu mô hình TFLite vào tệp đã chỉ định
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)
#
# print(f"Mô hình TFLite đã được lưu vào {tflite_model_path}")