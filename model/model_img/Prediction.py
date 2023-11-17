import cv2
import PIL.Image,PIL.ImageTk
from tensorflow import keras
# from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
import tensorflow  as tf
from keras.preprocessing import image
from numpy import argmax

cap=cv2.VideoCapture(0)

model=keras.models.load_model("lenet.h5", compile=False)
categories = ["cham", "hoa", "khmer", "kinh", "khac"] #thư mục

while True:
    ret , img=cap.read()
 
    image = img
    image = cv2.resize(image, (256, 256))

    image = np.array(image, dtype="float") / 255.0            
    image=np.expand_dims(image, axis=0)


    print(image.shape)
    pred=model.predict(image)
    Res=argmax(pred,axis=1)
    print(pred*100)
    print(Res)
    print(round(pred[0][Res[0]]*100,2))
    arg = list(np.argsort(pred, axis=1).flatten())[-2:]
    arg.sort()
    print(arg)
    if round(pred[0][Res[0]]*100,2)>80:
        Result_Text="{0}({1})".format(categories[Res[0]],round(pred[0][Res[0]]*100,2))
    else:
        Result_Text="Khong biet"
    cv2.putText(img,Result_Text, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
    
    cv2.imshow("camera",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


