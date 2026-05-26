import os
import random
import shutil

# ===================== 路径配置（不用改，直接用）=====================
IMAGE_TRAIN = "./images/train"    # 训练图片文件夹（里面有3746个子文件夹）
IMAGE_VAL   = "./images/val"      # 验证图片文件夹（自动创建）
LABEL_TRAIN = "./labels/train"    # 训练标签文件夹
LABEL_VAL   = "./labels/val"      # 验证标签文件夹（自动创建）

# 自动创建目标文件夹
os.makedirs(IMAGE_VAL, exist_ok=True)
os.makedirs(LABEL_VAL, exist_ok=True)

# ===================== 获取所有文件夹 =====================
# 读取 image/train 下的所有子文件夹名称
all_folders = [f for f in os.listdir(IMAGE_VAL)]
print(f"总文件夹数量：{len(all_folders)}")

# ===================== 开始移动 =====================
success = 0
for folder_name in all_folders:
    # ---------------- 2. 移动同名标签 .txt ----------------
    folder_name = folder_name.replace(".jpg", "")
    src_lbl = os.path.join(LABEL_TRAIN, f"{folder_name}.txt")
    dst_lbl = os.path.join(LABEL_VAL, f"{folder_name}.txt")
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

    folder_name = folder_name.replace(".jpeg", "")
    src_lbl = os.path.join(LABEL_TRAIN, f"{folder_name}.txt")
    dst_lbl = os.path.join(LABEL_VAL, f"{folder_name}.txt")
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

    success += 1

all_folders = [f for f in os.listdir(IMAGE_VAL)]
print(f"总文件夹数量：{len(all_folders)}")
all_folders = [f for f in os.listdir(LABEL_VAL)]
print(f"总文件夹数量：{len(all_folders)}")

print(f"\n✅ 移动完成！")
print(f"成功移动：{success} 个图片文件夹 + {success} 个标签文件")