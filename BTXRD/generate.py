import os

# 配置参数（根据你的实际路径调整）
DATA_DIR = "./BTXRD"  # 数据集根目录
IMAGE_TRAIN_DIR = os.path.join(DATA_DIR, "images/train")  # 训练图片目录
IMAGE_VAL_DIR = os.path.join(DATA_DIR, "images/val")      # 验证图片目录
TRAIN_TXT_PATH = os.path.join(DATA_DIR, "train.txt")      # 输出train.txt路径
VAL_TXT_PATH = os.path.join(DATA_DIR, "val.txt")          # 输出val.txt路径

# 支持的图片后缀（根据你的数据集调整）
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

def generate_file_list(img_dir, output_txt):
    """
    遍历图片目录，生成文件名清单txt
    :param img_dir: 图片目录路径
    :param output_txt: 输出txt文件路径
    """
    # 检查目录是否存在
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")
    
    # 遍历目录，筛选图片文件
    img_filenames = []
    for filename in os.listdir(img_dir):
        # 只保留图片文件，忽略隐藏文件/文件夹
        if filename.lower().endswith(SUPPORTED_EXT) and not filename.startswith("."):
            img_filenames.append(filename)
    
    # 排序（可选，保证顺序固定）
    img_filenames.sort()
    
    # 写入txt文件
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(img_filenames))
    
    print(f"成功生成 {output_txt}，共 {len(img_filenames)} 个样本")

if __name__ == "__main__":
    # 生成train.txt
    generate_file_list(IMAGE_TRAIN_DIR, TRAIN_TXT_PATH)
    # 生成val.txt
    # generate_file_list(IMAGE_VAL_DIR, VAL_TXT_PATH)