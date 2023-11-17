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

path = "D:/luan-van-data/img/cuoi/" #đường dẫm
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục


data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 128
WIDTH = 128
# 24 24
N_CHANNELS = 3

# ===========================lay ngau nhien anh===================================

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
plt.show()


# //////////////////////////////MobileNet////////////////////////////////

EPOCHS = 100
INIT_LR = 1e-3
BS = 14
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

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=10, shuffle=True)
# Chuyển đổi các nhãn lớp thành dạng one-hot encoding
trainY = to_categorical(trainY, len(class_names))

# Định nghĩa early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#Training
history = model.fit(trainX, trainY,validation_split=0.1, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])
pred = model.predict(testX)
predictions = argmax(pred, axis=1) # return to label

cm = confusion_matrix(testY, predictions)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + categories)
ax.set_yticklabels([''] + categories)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(i, j, cm[j, i], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

recall= recall_score(testY, predictions,average='weighted')
print("Recall: %.2f%%" % (recall * 100.0))

precision = precision_score(testY, predictions,average='weighted')
print("Precision: %.2f%%" % (precision * 100.0))

f1 = f1_score(testY, predictions,average='weighted')
print("F1 Score: %.2f%%" % (f1 * 100.0))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Xác định đường dẫn tệp mà bạn muốn lưu mô hình TFLite
tflite_model_path = "mobile_net_cuoi.tflite"

# Lưu mô hình TFLite vào tệp đã chỉ định
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Mô hình TFLite đã được lưu vào {tflite_model_path}")