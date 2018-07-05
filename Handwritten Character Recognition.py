# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 19:56:22 2018

@author: LWC
"""

import numpy as np  # 匯入 Numpy 套件
import pandas as pd  # 匯入 Pandas 套件
import matplotlib.pyplot as plt  # 匯入繪圖套件
from keras.datasets import mnist  # 匯入 MNIST 後載入資料集
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout  # 以避免 Overfitting
from keras.layers import Flatten  # 將二維矩陣轉換為一維向量
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils  # 匯入 Keras 的 Numpy 函數，將標籤轉為 one-hot-encoding
from keras import backend as K
K.set_image_dim_ordering('th')

# 設定隨機種子，使每次執行結果相同
seed = 7
np.random.seed(seed)

print("===================================STEP 1. 讀取與查看 MNIST 數據庫===================================")
# 未下載則自動下載，已下載則讀取
(X_train_image, Y_train_label), (X_test_image, Y_test_label) = mnist.load_data()
print("[Info] train data={:7,}".format(len(X_train_image)))
print("[Info] test  data={:7,}".format(len(X_test_image)))

print("\n")

print("=========================================STEP 2. 查看訓練資料=========================================")
print("[Info] Shape of train data=%s" % (str(X_train_image.shape)))
print("[Info] Shape of train label=%s" % (str(Y_train_label.shape)))

# 定義 plot_image 函數顯示數字影像
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')  # cmap='binary' 參數設定以黑白灰階顯示
    plt.show()


plot_image(X_train_image[0])
print("This image's label is", Y_train_label[0])

print("\n")

print("=================================STEP 3. 查看多筆訓練資料的圖形和標籤=================================")


def plot_images_labels_predict(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "l="+str(labels[idx])
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
        else:
            title = "l={}".format(str(labels[idx]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# 查看訓練資料的前 10 筆資料
plot_images_labels_predict(X_train_image, Y_train_label, [], 0, 10)

print("\n")

print("=========================================STEP 4. 資料預處理=========================================")
# 將 image 以 reshape 轉換為二維 ndarray : [samples][pixels][width][height]
X_Train = X_train_image.reshape(
    X_train_image.shape[0], 1, 28, 28).astype('float32')
X_Test = X_test_image.reshape(
    X_test_image.shape[0], 1, 28, 28).astype('float32')

print(" xTrain: %s" % (str(X_Train.shape)))
print(" xTest: %s" % (str(X_Test.shape)))

# 把輸入從 0-255 標準化到 0-1
X_Train_norm = X_Train/255
X_Test_norm = X_Test/255

# 進行 One-hot-encoding 呈現輸出
Y_Train_OneHot = np_utils.to_categorical(Y_train_label)
Y_Test_OneHot = np_utils.to_categorical(Y_test_label)
num_classes = Y_Test_OneHot.shape[1]

print("\n")

print("==========================================STEP 5. 建立模型==========================================")
model = Sequential()  # 建立線性模型
model.add(Conv2D(30, 5, 5, border_mode='valid',
                        input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=1000, input_dim=784,
                kernel_initializer='normal', activation='relu'))  # 添加輸入/第一個隱藏層
model.add(Dropout(0.5))
model.add(Dense(units=1000, kernel_initializer='normal',
                activation='relu'))  # 添加第二個隱藏層
model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='normal',
                activation='softmax'))  # 添加隱藏/輸出層
print("[Info] Model summary:")
model.summary()
print("")

print("\n")

print("============================================STEP 6. 訓練============================================")
# 定義訓練方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])  # 編譯模型

# 開始訓練
train_history = model.fit(X_Train_norm, Y_Train_OneHot,
                          validation_split=0.2, epochs=15, batch_size=200, verbose=2)

# 建立 show_train_history 顯示訓練過程
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# 評估模型準確率
scores = model.evaluate(X_Test_norm, Y_Test_OneHot)

print("[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

print("\n")

print("================================STEP 7. 以測試資料評估模型準確率與預測================================")
# 進行預測
print("[Info] Making prediction to X_Test_norm")
prediction = model.predict_classes(X_Test_norm)  # 進行預測並把結果儲存到預測中
print()
print("[Info] Show 10 prediction result (From 240):")
print("%s\n" % (prediction[240:250]))

plot_images_labels_predict(X_test_image, Y_test_label, prediction, idx=240)

print("[Info] Error analysis:")
for i in range(len(prediction)):
    if prediction[i] != Y_test_label[i]:
        print("At %d'th: %d is with wrong prediction as %d!" %
              (i, Y_test_label[i], prediction[i]))

print("\n")

print("========================================STEP 8. 顯示混淆矩陣========================================")
print("[Info] Display Confusion Matrix:")
print("%s\n" % pd.crosstab(Y_test_label, prediction,
                           rownames=['label'], colnames=['predict']))
