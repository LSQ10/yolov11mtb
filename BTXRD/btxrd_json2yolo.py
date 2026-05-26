import os
import json
import glob

# ====================== 配置 ======================
JSON_DIR = "./BTXRD/Annotations"  # 存放所有 .json 的文件夹
IMAGE_DIR = "./BTXRD/images"      # 存放所有图片的文件夹
OUTPUT_DIR = "./BTXRD/labels"      # 输出 .txt 的文件夹
CLASS_MAP = {
    "tumor": 0,
    "other mt": 1,
    "other bt": 2,
    "osteosarcoma": 3,
    "osteochondroma": 4,
    "simple bone cyst": 5,
    "giant cell tumor": 6,
    "osteofibroma": 7,
}
# ===================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(JSON_DIR, json_name)
    txt_path = os.path.join(OUTPUT_DIR, json_name.replace(".json", ".txt"))

    # 无json → 空txt（正常骨骼）
    if not os.path.exists(json_path):
        with open(txt_path, "w") as f:
            f.write("")
        continue

    # 有json → 转换
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_w = data["imageWidth"]
        img_h = data["imageHeight"]
        lines = []

        for shape in data["shapes"]:
            label = shape["label"].strip().lower()
            shape_type = shape["shape_type"]
            points = shape["points"]

            if shape_type == "rectangle":
                x1, y1 = points[0]
                x2, y2 = points[1]
                x = (x1 + x2) / 2 / img_w
                y = (y1 + y2) / 2 / img_h
                w = abs(x2 - x1) / img_w
                h = abs(y2 - y1) / img_h

                if label in CLASS_MAP:
                    cls_id = CLASS_MAP[label]
                    lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    except Exception as e:
        print(f"错误 {json_name}: {e}")
        with open(txt_path, "w") as f:
            f.write("")

print(f"✅ 转换完成！共生成 {len(glob.glob(os.path.join(OUTPUT_DIR, '*.txt')))} 个标签")