import os
import shutil
from read_pgm import read_pgm
from keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt

# Khai bao bien mac dinh
TEMP = 1

# Tao thu muc validation
if not os.path.exists("dataset/validation"):
    os.makedirs("dataset/validation")
if not os.path.exists("dataset/train"):
    os.makedirs("dataset/train")

# Doc du lieu
data_path = "att_faces"
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue

    # Tao thu muc, kiem tra
    result_path1 = "dataset/train/{}".format(dir_name)
    if not os.path.exists(result_path1):
        os.makedirs(result_path1)
    result_path2 = "dataset/validation/{}".format(dir_name)
    if not os.path.exists(result_path2):
        os.makedirs(result_path2)

    #Lay duong dan file
    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        img = read_pgm(file_path)
        file_path_new = file_path + '.jpg'
        cv2.imwrite(file_path_new, img)

        if TEMP <= 2:
            shutil.move(file_path_new, result_path2)
            TEMP += 1
        else:
            shutil.move(file_path_new, result_path1)
    TEMP = 1