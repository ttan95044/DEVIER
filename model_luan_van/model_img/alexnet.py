import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import tensorflow.keras as keras
from keras.layers import Dense, Flatten
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

for imagePath in imagePaths:
    if not os.path.isfile(imagePath[0]):
        print("Error: File not found -", imagePath[0])
        continue
    image = cv2.imread(imagePath[0])
    if image is None:
        print("Error: Unable to read image -", imagePath[0])
        continue

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    if image is None:
        print("Error: Unable to read image -", imagePath[0])
        continue
    if image.size == 0:  # Kiểm tra kích thước ảnh
        print("Error: Empty image -", imagePath[0])
        continue
    image = cv2.resize(image, (WIDTH, HEIGHT))  # Thay đổi kích thước ảnh
    data.append(image)
    label = imagePath[1]
    labels.append(label)

# #đọc và xử lý hình ảnh từ danh sách imagePaths
# for imagePath in imagePaths:#duyệt qua từng imagepath
#     image = cv2.imread(imagePath[0])#đọc ảnh từ đường dẫn
#     image = cv2.resize(image, (WIDTH, HEIGHT))  # thay đổi kích thước ảnh
#     data.append(image)# thêm vào danh sách data
#     label = imagePath[1] #gán nhãn cho dữ liệu tương ứng
#     labels.append(label)

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

#Alexnet Architcture
model = keras.Sequential()

model.add(Conv2D(filters= 96 , kernel_size=(11,11) ,strides= 4, padding='valid',  activation= 'relu', input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPool2D(pool_size=(3,3) , strides= 2 ))

model.add(Conv2D(filters= 256 , kernel_size=(5,5) ,strides= 1 ,padding='same' , activation= 'relu'))
model.add(MaxPool2D(pool_size=(3,3) , strides= 2 ))

model.add(Conv2D(filters= 384 , kernel_size=(3,3) ,strides= 1 ,padding='same' , activation= 'relu'))
model.add(Conv2D(filters= 384 , kernel_size=(3,3) ,strides= 1 ,padding='same' , activation= 'relu'))
model.add(Conv2D(filters= 256 , kernel_size=(3,3) ,strides= 1 ,padding='same' , activation= 'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(9216 , activation='relu'))
model.add(Dense(4096 , activation='relu'))
model.add(Dense(4096 , activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

optim = optimizers.Adam()
model.compile(optim , loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

results1 = []
results2 = []
results3 = []
results4 = []

for e in range(1,2):
    start_time = time.time()

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=15, shuffle=True)

    # Chuyển đổi các nhãn lớp thành dạng one-hot encoding
    trainY = to_categorical(trainY, len(class_names))
    testY = to_categorical(testY, len(class_names))

    # Định nghĩa early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()  # Bắt đầu đo thời gian kiểm tra
    # đoạn này dùng để save mô hình
    # model.save("lenet.h5")

    # đoạn này dùng để kiểm tra mô hình

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
print("Trung bình của 10 lần huấn luyện accuracy:", average_accuracy)

average_recall = sum(results2) / len(results2)
print("Trung bình của 10 lần huấn luyện recall:", average_recall)

average_precision = sum(results3) / len(results3)
print("Trung bình của 10 lần huấn luyện precision:", average_precision)

average_f1 = sum(results4) / len(results4)
print("Trung bình của 10 lần huấn luyện f1:", average_f1)
