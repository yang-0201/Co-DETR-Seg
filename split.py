import json
import numpy as np
from sklearn.model_selection import train_test_split

# 加载原始COCO标注文件
with open('/home/data/Co-DETR/train/train_annotations.json', 'r') as f:
    coco_data = json.load(f)

# 提取关键字段
images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

# 随机打乱images并划分 (9:1)
image_ids = [img['id'] for img in images]
train_ids, val_ids = train_test_split(image_ids, test_size=0.1, shuffle=True, random_state=42)

# 根据划分的image_id提取训练集和验证集数据
def filter_annotations(all_annotations, image_ids):
    return [ann for ann in all_annotations if ann['image_id'] in image_ids]

train_ann = filter_annotations(annotations, train_ids)
val_ann = filter_annotations(annotations, val_ids)

# 构建训练集和验证集COCO结构
train_data = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "categories": categories,
    "images": [img for img in images if img['id'] in train_ids],
    "annotations": train_ann
}

val_data = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "categories": categories,
    "images": [img for img in images if img['id'] in val_ids],
    "annotations": val_ann
}

# 保存文件
with open('train.json', 'w') as f:
    json.dump(train_data, f)

with open('val.json', 'w') as f:
    json.dump(val_data, f)

print(f"划分完成！训练集: {len(train_ids)}张, 验证集: {len(val_ids)}张")