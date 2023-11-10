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
from tensorflow.keras.applications import InceptionV3,MobileNet,VGG16,DenseNet121,EfficientNetB7#DenseNet,EfficientNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from keras import models
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from keras.callbacks import EarlyStopping
import time

path = "D:/luan-van-data/sound_img/" #đường dẫm
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục



data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 128
WIDTH = 128
# 24 24
N_CHANNELS = 3

# ===========================lay ngau nhien anh===================================

# Duyệt qua danh mục và tạo danh sách đường dẫn
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

# =======================tien xu ly=======================================

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

plt.subplots(3,4)
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
# plt.show()


# //////////////////////////////MobileNet////////////////////////////////

EPOCHS = 100
INIT_LR = 1e-3
BS = 64
#--------------------------------------------
class_names = categories
#--------------------------------------------




print("[INFO] compiling model...")
mobileNet = MobileNet(input_shape=(WIDTH, HEIGHT, N_CHANNELS), include_top=False, weights='imagenet')
for layer in mobileNet.layers:
    layer.trainable = False

model = Sequential()
model.add(mobileNet)
#model.add(layers.AveragePooling2D((8, 8), padding='valid', name='avg_pool'))
model.add(GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(len(class_names), activation='softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

results1 = []
results2 = []
results3 = []
results4 = []

for e in range(1,2):
    start_time = time.time()
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

    # Chuyển đổi các nhãn lớp thành dạng one-hot encoding
    trainY = to_categorical(trainY, len(class_names))

    # Định nghĩa early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    #Training
    history = model.fit(trainX, trainY,validation_split=0.2, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
    end_time = time.time()  # Kết thúc đo thời gian huấn luyện

    start_time_test = time.time()

    pred = model.predict(testX)
    predictions = argmax(pred, axis=1) # return to label

    # cm = confusion_matrix(testY, predictions)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Model confusion matrix')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + categories)
    # ax.set_yticklabels([''] + categories)
    #
    # for i in range(len(class_names)):
    #     for j in range(len(class_names)):
    #         ax.text(i, j, cm[j, i], va='center', ha='center')
    #
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    end_time_test = time.time()  # Kết thúc đo thời gian kiểm tra
    training_time = end_time - start_time
    testing_time = end_time_test - start_time_test
    print("Thời gian huấn luyện:", training_time, "giây")
    print("Thời gian kiểm tra:", testing_time, "giây")

    accuracy = accuracy_score(testY, predictions)
    # print("Accuracy : %.2f%%" % (accuracy * 100.0))
    results1.append(accuracy)

    recall= recall_score(testY, predictions,average='weighted')
    results2.append(recall)
    # print("Recall: %.2f%%" % (recall * 100.0))

    precision = precision_score(testY, predictions,average='weighted')
    results3.append(precision)
    # print("Precision: %.2f%%" % (precision * 100.0))

    f1 = f1_score(testY, predictions,average='weighted')
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