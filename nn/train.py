
import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2
from sklearn.utils import shuffle

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense

#Đọc dữ liệu
data_path = "att_faces"
x_train = []
y_train = []
x_test = []
y_test = []
label = 0
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue
    if label >= 40:
        break

    value = 0
    #Lay duong dan file
    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
        img = img / 255.0  # Chuẩn hóa 0-255 thành 0-1
        img = cv2.resize(img, (92, 112))

        if value < 8:
            x_train.append(img)
            y_train.append(label)
            value += 1
        else:
            x_test.append(img)
            y_test.append(label)
            value += 1
    label += 1
print(y_train)
#Format
x_train, y_train = shuffle(x_train, y_train)
x_train = np.reshape(x_train, [-1, 92, 112, 1])
y_train = to_categorical(y_train)
x_test = np.reshape(x_test, [-1, 92, 112, 1])
y_test = to_categorical(y_test)

# Build model
model = Sequential()

# add model layers
model.add(Flatten(input_shape=(92, 112, 1)))
model.add(Dense(235, activation='relu'))
model.add(Dense(40, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Tao thu muc models
path = "models/train1"
if not os.path.exists(path):
    os.makedirs(path)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Save model
with open(path + '/report.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

model_json = model.to_json()
with open(path + "/face_model.json", "w") as json_file:
    json_file.write(model_json)
with open(path + "/face_model_weights.h5", "w") as json_file:
    model.save_weights(path +'/face_model_weights.h5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path + '/acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path + '/loss.png')
plt.show()