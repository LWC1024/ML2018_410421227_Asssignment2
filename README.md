# Handwritten Character Recognition
Machine Learning - Programming Asssignment2
## A. The way i prepare the training samples
  嘗試了 MLP ，但因 CNN 在圖像辨識上有明顯的優勢，最後選擇使用 CNN 完成
## B. All parameters i used for the training algorithm
dim_ordering = th  
```
輸入數據格式為[samples][channels][rows][cols]  
```
channels = 1  
```
灰階圖片  
```
seed 隨機種子 = 7  
```
使每次執行結果相同  
```
**model.add - Convolution2D 2D卷積層 :**  
filter 過濾器 = 30個  
row 過濾器行數 = 5  
col 過濾器列數 = 5  
border_mode 採集圖片邊緣特徵的模式 = valid  
activation 激勵函數 = relu

**model.add - MaxPooling2D 匯集層 :**  
pool_size 窗口大小 = 2*2  
**model.add - Dropout = 0.2**  
```
隨機踢出的機率  
```
**model.add - Flatten 平化層**  
```
將二維陣列轉換為一維陣列  
```
**model.add - Dense 全連接層 (隱藏層) :**  
input_dim 輸入節點數量 = 784個  
init 權重初始化 = normal  
activation 激勵函數 = relu

**model.add - Dense 全連接層 (輸出層) :**  
init 權重初始化 = normal  
activation 激勵函數 = softmax

**model.compile :**  
loss 誤差 = categorical_crossentropy  
```
即為 Logarithmic 函數  
```
optimizer 優化器 = adam  
```
ADAM 梯度下降法  
```
**model.fit :**  
epochs 訓練週期 = 10  
batch_size = 200  
```
每處理200個圖片進行一次權重更新  
```
verbose = 2  
```
每個訓練週期完成後只輸出一條日誌  
```
## C. Model summary
![](https://github.com/LWC1024/ML2018_410421227_Asssignment2/blob/master/result/model.jpg "CNN 模型")
## D. Training result
![](https://github.com/LWC1024/ML2018_410421227_Asssignment2/blob/master/result/result.jpg "訓練過程")  
![](https://github.com/LWC1024/ML2018_410421227_Asssignment2/blob/master/result/acc_99.2%25.png "ACC")  
![](https://github.com/LWC1024/ML2018_410421227_Asssignment2/blob/master/result/loss_2.52%25.png "LOSS")
## E. Confusion matrix
![](https://github.com/LWC1024/ML2018_410421227_Asssignment2/blob/master/result/confusion%20matrix.jpg "混淆矩陣")
## F. The problems i encountered
* ACC成果不理想
* overfitting
## G. I learned from this work
```
也許是因為已經有了一次的經驗，加上上學期上過人工智慧，所以這次Github的操作和MNIST的訓練順利很多，少數幾個癥結點是一開始參數設定不太理想，導致結果的準確度不太高，經過多次實驗，加上上網找一些文章，一步一腳印的訓練出這次滿意的結果，CNN真的無所不在，希望還能再學多一些。
```
