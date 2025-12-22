import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import l2

# ** 1. 設定超參數**
IMG_SIZE = (128, 128)  # 影像大小
BATCH_SIZE = 32  # 批次大小
EPOCHS = 40  # 訓練回合數
LEARNING_RATE = 0.0001  # 學習率

# ** 2. 設定資料夾路徑**
train_dir = "dataset/train"
test_dir  = "dataset/test"
SAVE_PATH = "model/hand_gesture_model.h5"
SAVE_PICT = "assets/training_result.png"

# ** 3. 數據增強**
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # 旋轉範圍擴大到 15 度
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.05,
    zoom_range=0.2,  # 加大 zoom_range 以增強適應性
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ** 4. 載入訓練 & 測試資料**
print(" 正在載入訓練資料...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print(" 正在載入測試資料...")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ** 5. 確保標籤順序正確**
num_classes = len(train_generator.class_indices)
print(f" 類別數量（num_classes）：{num_classes}")
print(" 確認 class_indices（標籤）：")
print(train_generator.class_indices)

# ** 6. 使用 MobileNetV2 作為特徵提取器**
print(" 載入 MobileNetV2 預訓練模型...")
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # **凍結預訓練權重，不參與訓練**

# ** 7. 建立自訂分類層**
print(" 建立自訂分類層...")
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  # 加入 L2 正則化
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ** 8. 設定編譯參數**
print(" 編譯模型...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ** 9. 加入 ReduceLROnPlateau (學習率調整)**
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

# ** 10. 訓練模型 (第一階段)**
print(" 開始訓練模型 (預訓練權重凍結)...")
history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[reduce_lr])

# ** 11. Fine-Tuning: 解凍 MobileNetV2 最後 30 層**
print(" 解凍 MobileNetV2 最後 30 層進行微調...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # 只解凍最後 30 層
    layer.trainable = False

# ** 12. 重新編譯模型 (較低學習率)**
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ** 13. 訓練模型 (第二階段)**
print(" 開始微調模型...")
history_finetune = model.fit(train_generator, epochs=30, validation_data=test_generator, callbacks=[reduce_lr])

# ** 14. 儲存模型**
print(f" 儲存模型至：{SAVE_PATH}")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)

# ** 15. 評估模型**
print(" 評估模型準確率...")
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)

print(f"\n Train Accuracy: {train_acc * 100:.2f}%")
print(f" Test Accuracy: {test_acc * 100:.2f}%\n")

# ** 16. 產生預測 & 分類報告**
print(" 產生分類報告...")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
print(report)

# ** 17. 顯示最終結果**
print("\n **最終訓練結果** ")
print(f"Loss: {history_finetune.history['loss'][-1]:.4f}")
print(f"Accuracy: {history_finetune.history['accuracy'][-1]:.4f}")
print(f"Val Loss: {history_finetune.history['val_loss'][-1]:.4f}")
print(f"Val Accuracy: {history_finetune.history['val_accuracy'][-1]:.4f}\n")

print(f" Train Accuracy: {train_acc * 100:.2f}%")
print(f" Test Accuracy: {test_acc * 100:.2f}%")
print(f" Train Loss: {train_loss:.4f}")
print(f" Test Loss: {test_loss:.4f}\n")

# ** 18. 計算 Precision, Recall, F1-score**
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]

print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1-score: {f1_score:.4f}")

# ** 19. 繪製訓練過程**
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.tight_layout()
plt.savefig(SAVE_PICT)
print(f" 圖表已儲存為：{SAVE_PICT}")

plt.show()

print(" 訓練與測試完成！")
