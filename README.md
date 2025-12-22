# HandGestureRecognition

A **Taiwan-style hand gesture digit recognition** system (0–9) built with **TensorFlow + MobileNetV2**.  
使用 **TensorFlow + MobileNetV2** 實作「數字手勢辨識（0~9）」：從自行拍攝資料、前處理、訓練到推論的完整流程。

---

## Overview / 專案簡介

- **Task**：影像分類（0~9 十類）
- **Model**：MobileNetV2 (ImageNet pre-trained) + Custom Classifier
- **Input**：128×128 RGB
- **Goal**：辨識台灣手勢數字 0~9（左右手、正反面）

---

## Environment / 環境

> 本專案原本於下列環境完成訓練（可依你的設備調整）：

- WSL2 Ubuntu 22.04
- Python 3.10
- TensorFlow 2.15.0
- CUDA 12.2 + cuDNN 8.9.7 (GPU)

---

## Dataset / 資料集

本資料集為**自行拍攝**（左右手 + 正反面），分類為 **0~9** 共 10 類：

- **資料數量**：每個數字 200 張  
  - 左手正面 50、左手反面 50、右手正面 50、右手反面 50
- **圖片格式**：JPG / RGB
- **解析度**：統一調整為 128×128
- **資料分組**：以資料夾分群，例如：
  - `dataset/train/0/`, `dataset/train/1/` ... `dataset/train/9/`
- **切分方式**：80% 訓練、20% 驗證  
  - 每類：160 張 train、40 張 test  
  - 總計：1600 張 train、400 張 test

---

## Data Augmentation / 數據增強

使用 `ImageDataGenerator`：

**Train**
- rescale=1./255
- rotation_range=15
- width_shift_range=0.2
- height_shift_range=0.2
- shear_range=0.05
- zoom_range=0.2
- horizontal_flip=True

**Test**
- 僅 rescale=1./255（不做增強）

---

## Model Architecture / 模型架構

使用 **MobileNetV2** 作為特徵提取器（`include_top=False`, `weights='imagenet'`）：

```text
Input (128x128x3)
   ↓
MobileNetV2 (pretrained, include_top=False)
   ↓
GlobalAveragePooling2D
   ↓
Dense(256, ReLU) + L2 regularization (l2=0.001)
   ↓
Dropout(0.5)
   ↓
Dense(10, Softmax)
   ↓
Output (classes: 0–9)
```

---

```md
## Results / 實驗結果

訓練過程總結：
- Accuracy: 0.9125
- Val Accuracy: 0.9400
- Loss: 0.6431
- Val Loss: 0.5602

最終表現：
- Train Accuracy: **94.19%**
- Test Accuracy: **94.00%**
- Train Loss: 0.5318
- Test Loss: 0.5602

評估指標:
- Precision: 0.9478
- Recall: 0.9400
- F1-score: 0.9399
```
### Training Curve (Accuracy / Loss)
<img src="assets/training_result.png" width="800"/>

> The curve shows stable accuracy improvement and decreasing loss without obvious overfitting.

---

## Project Structure / 專案結構

```text
HandGestureRecognition/
│
├─ assets/
│  └─ training_result.png
│
├─ model/
│  └─ hand_gesture_model.h5            # (optional) trained model
│
├─ src/
│  ├─ hand_gesture_build_model.py      # training script
│  ├─ hand_gesture_predict.py          # inference script
│  └─ preprocess/
│     ├─ segmentation.py               # split train/test (80/20)
│     └─ rename_images.py              # rename images to gesture_{k}_{idx}.jpg
│
├─ dataset/                            #(Excluded from Git) 0-9 gesture images
│  ├─ train/
│  │  ├─ 0/ ... 9/
│  └─ test/
│     ├─ 0/ ... 9/
│
└─ README.md
```

