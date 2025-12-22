import os

# 設定手勢數字（0~9）
gesture_number = 9 # 你可以改成 0,1,2,... 來處理不同手勢

# 設定圖片資料夾
folder_path = r"dataset\test\9"

# **使用 os.scandir() 確保讀取的是最新檔案清單**
file_list = sorted([entry.name for entry in os.scandir(folder_path) if entry.is_file()],
                   key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))  

# 確保圖片數量足夠
if len(file_list) < 40:
    print(f" 錯誤：{folder_path} 內的圖片少於 40 張！")
    exit()

# **重新命名所有圖片**
for idx, filename in enumerate(file_list[:40]):  # 只處理前 40 張
    old_path = os.path.join(folder_path, filename)
    
    # **確保檔案存在，避免 FileNotFoundError**
    if not os.path.isfile(old_path):
        print(f" 找不到檔案：{old_path}，跳過")
        continue

    # 設定新檔名（不分手勢類型）
    new_name = f"gesture_{gesture_number}_{idx+1:03d}.jpg"
    new_path = os.path.join(folder_path, new_name)

    # **確保新檔名不會覆蓋舊檔案**
    if os.path.exists(new_path):
        print(f" 檔案已存在：{new_path}，跳過")
        continue

    # **執行重新命名**
    try:
        os.rename(old_path, new_path)
        print(f" {old_path} -> {new_path}")
    except Exception as e:
        print(f" 重新命名失敗：{old_path} -> {new_path}, 錯誤：{e}")
        continue

print(" 批次命名完成！")
