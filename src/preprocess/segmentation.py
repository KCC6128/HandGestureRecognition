import os
import random
import shutil

# 設定資料夾路徑
train_dir = r"dataset\train"
test_dir  = r"dataset\test"

# 設定測試集比例（20%）
test_ratio = 0.2

# 確保 test 資料夾存在
os.makedirs(test_dir, exist_ok=True)

# 逐一處理 0~9 的手勢資料夾
for gesture_number in range(10):  
    gesture_train_dir = os.path.join(train_dir, str(gesture_number))  
    gesture_test_dir = os.path.join(test_dir, str(gesture_number))  

    # 確保 test 子資料夾存在
    os.makedirs(gesture_test_dir, exist_ok=True)

    # 取得所有訓練圖片
    all_images = [f for f in os.listdir(gesture_train_dir) if os.path.isfile(os.path.join(gesture_train_dir, f))]

    # 計算要移動的圖片數量（20%）
    num_test_images = int(len(all_images) * test_ratio)

    # 隨機選擇要移動的圖片
    test_images = random.sample(all_images, num_test_images)

    # 移動圖片到 test 資料夾
    for image in test_images:
        src_path = os.path.join(gesture_train_dir, image)
        dest_path = os.path.join(gesture_test_dir, image)
        shutil.move(src_path, dest_path)
        print(f" 移動 {src_path} -> {dest_path}")

print(" 測試集分割完成！")
