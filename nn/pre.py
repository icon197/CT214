from __future__ import absolute_import, division, print_function

# Thêm thư viện TensorFlow và tf.keras
import tensorflow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.metrics import classification_report
#Thư viện open file
from tkinter import filedialog
from tkinter import *

# Load json và tạo model
if __name__ == "__main__":

    # Load model age
    json_file = open("models/train235/face_model.json", "r")
    json_string = json_file.read()
    json_file.close()
    model = model_from_json(json_string)
    model.load_weights("models/train235/face_model_weights.h5")
    names = ["S33", "S23", "S15", "S13", "S34", "S32", "S31", "S24", "S38", "S12", "S20",
             "S35", "S18", "S3", "S17", "S16", "S14", "S4", "S40", "S30", "S39", "S26",
             "S5", "S10", "S27", "S28", "S8", "S25", "S1", "S37", "S21", "S22", "S7",
             "S29", "S2", "S19", "S6", "S9", "S36", "S11"
             ]

    #Chọn file ảnh
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir = "/",title = "Chọn file ảnh",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    # Xử lý ảnh đầu vào
    img = cv2.imread(root.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    img = img / 255.0  # Chuẩn hóa 0-255 thành 0-1
    img = cv2.resize(img, (92, 112))
    plt.imshow(img)
    plt.show()

    # #Tiền sử lý
    img_arr = img.reshape(-1, 92, 112, 1)

    #Dự đoán
    prediction = model.predict(img_arr)
    print(prediction)
    print(np.argmax(prediction[0]))
    print(names[np.argmax(prediction[0])])

    root.destroy()