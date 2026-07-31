"""
CLIP 脑部CT中风分类
图像文本分类 · 零样本学习
"""

import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# ── 设备 ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device.upper()}")

# ── 加载CLIP模型 ──
print("加载CLIP模型中...")
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"✅ CLIP加载完成 (参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

# ── 文字描述模板（中医学术语） ──
STROKE_PROMPTS = [
    "这张脑部CT片显示急性中风，有高密度病灶区",
    "脑部CT可见左侧基底节区高密度影，提示急性脑出血",
    "CT显示大脑中动脉供血区低密度灶，提示急性脑梗死",
    "头部CT平扫可见右侧额颞叶低密度梗死灶",
    "脑部CT显示高密度血肿，符合急性脑出血表现",
    "CT影像可见明显的缺血性改变，符合中风诊断",
    "大脑半球可见片状低密度区，提示急性脑卒中",
    "CT显示脑实质内高密度灶，周围有水肿带",
    "这是一张急性中风患者的脑部CT扫描图像",
    "脑部CT可见基底节区出血，中线结构无明显移位",
]

NORMAL_PROMPTS = [
    "这张脑部CT片显示正常，无异常密度影",
    "脑部CT平扫未见明显异常密度灶",
    "头部CT显示脑实质密度均匀，灰白质分界清晰",
    "CT影像未见脑出血、脑梗死或占位性病变",
    "脑部CT显示正常的脑室系统和脑沟回结构",
    "这是一张正常成年人的脑部CT扫描图像",
    "CT显示双侧大脑半球对称，无异常密度改变",
    "脑部CT平扫未见新鲜病灶，中线结构居中",
    "头部CT显示脑实质未见明确异常密度影",
    "CT影像显示正常脑部解剖结构，无异常发现",
]

# 编码文字
class_texts = ["一张显示急性中风的脑部CT片", "一张显示正常的脑部CT片"]
class_tokens = clip.tokenize(class_texts).to(device)
with torch.no_grad():
    class_features = model.encode_text(class_tokens)
    class_features /= class_features.norm(dim=-1, keepdim=True)
print("✅ 文字描述编码完成")

# ── 数据集 ──
class CTDataset(Dataset):
    def __init__(self, data_dir, split="test", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        for fold in ["Fold1", "Fold2", "Fold3"]:
            for cls_name, cls_idx in [("Stroke", 0), ("Normal", 1)]:
                cls_dir = os.path.join(data_dir, fold, split, cls_name)
                if not os.path.exists(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(cls_dir, fname))
                        self.labels.append(cls_idx)

        print(f"  共加载 {len(self.images)} 张图片")
        print(f"  中风: {sum(1 for l in self.labels if l==0)} 张")
        print(f"  正常: {sum(1 for l in self.labels if l==1)} 张")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ── 加载数据 ──
DATA_DIR = "data/stroke-dataset"
print(f"\n加载数据: {DATA_DIR}")
dataset = CTDataset(DATA_DIR, split="test", transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ── CLIP零样本分类 ──
print("\n开始CLIP分类...")
all_preds, all_labels, all_probs = [], [], []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="分类中"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ class_features.T)
        probs = similarity.softmax(dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ── 结果 ──
acc = accuracy_score(all_labels, all_preds)
print(f"\n{'='*40}")
print(f"🎯 准确率: {acc:.2%}")
print(f"{'='*40}")

print(f"\n分类报告:")
print(classification_report(all_labels, all_preds,
      target_names=["中风(Stroke)", "正常(Normal)"]))

# ── 混淆矩阵 ──
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["预测中风", "预测正常"],
            yticklabels=["实际中风", "实际正常"])
plt.title(f"CLIP脑部CT中风分类 (准确率: {acc:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\n✅ 结果已保存: confusion_matrix.png")

# ── 置信度分布 ──
stroke_probs = all_probs[all_labels == 0, 0]
normal_probs = all_probs[all_labels == 1, 1]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(stroke_probs, bins=20, alpha=0.7, color='red')
plt.axvline(x=0.5, color='gray', linestyle='--')
plt.xlabel('预测中风概率')
plt.ylabel('图片数')
plt.title(f'中风图片置信度 (n={len(stroke_probs)})')

plt.subplot(1, 2, 2)
plt.hist(normal_probs, bins=20, alpha=0.7, color='green')
plt.axvline(x=0.5, color='gray', linestyle='--')
plt.xlabel('预测正常概率')
plt.ylabel('图片数')
plt.title(f'正常图片置信度 (n={len(normal_probs)})')

plt.tight_layout()
plt.savefig("confidence_distribution.png")
print("✅ 结果已保存: confidence_distribution.png")
print(f"\n🎉 项目完成！")
