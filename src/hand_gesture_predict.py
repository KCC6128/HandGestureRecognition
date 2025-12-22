import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

font_path = r"C:\Windows\Fonts\msjh.ttc"  # Windows 原生最常見
font_prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None


#  載入模型
model_path = "model/hand_gesture_model.h5"
model = tf.keras.models.load_model(model_path)

#  圖片路徑
img_path = "dataset/test/3/gesture_3_040.jpg"

#  類別標籤
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#  圖片預處理
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (128, 128))
img_normalized = img_resized / 255.0
img_batch = np.expand_dims(img_normalized, axis=0)

#  模型預測
predictions = model.predict(img_batch)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

#  印出預測結果（終端機）
print(f"預測結果：類別 {predicted_class}（手勢為：{class_labels[predicted_class]}，信心值：{confidence:.2f}）")

#  顯示圖片，並使用你要的格式作為標題
title_text = f"預測結果：{predicted_class}（手勢為：{class_labels[predicted_class]}，信心值：{confidence:.2f}）"

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(title_text, fontproperties=font_prop) if font_prop else plt.title(title_text)
plt.axis('off')
plt.show()
