"""
Kaggle Notebook: CLIP 脑部CT中风分类
直接复制到你的Kaggle笔记本里运行
"""

# ═══════════════════════════════════════════════
# Cell 1: 安装依赖（运行一次即可）
# ═══════════════════════════════════════════════

# !pip install git+https://github.com/openai/CLIP.git
# !pip install matplotlib seaborn scikit-learn

# ═══════════════════════════════════════════════
# Cell 2: 导入库
# ═══════════════════════════════════════════════

import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ═══════════════════════════════════════════════
# Cell 3: 加载CLIP模型
# ═══════════════════════════════════════════════

model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP模型参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ═══════════════════════════════════════════════
# Cell 4: 文字描述模板
# ═══════════════════════════════════════════════

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

# 编码所有文字描述
all_texts = STROKE_PROMPTS + NORMAL_PROMPTS
text_tokens = clip.tokenize(all_texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 每个类别的平均特征（类别级）
class_texts = ["一张显示急性中风的脑部CT片", "一张显示正常的脑部CT片"]
class_tokens = clip.tokenize(class_texts).to(device)

with torch.no_grad():
    class_features = model.encode_text(class_tokens)
    class_features /= class_features.norm(dim=-1, keepdim=True)

print(f"✅ 文字编码完成: {len(all_texts)}条描述")
print(f"   类别标签: {len(class_texts)}个")
print(f"   特征维度: {text_features.shape[-1]}")

# ═══════════════════════════════════════════════
# Cell 5: 加载数据
# ═══════════════════════════════════════════════
# ⚠️ 请根据你的实际数据集路径修改！

DATA_DIR = "/kaggle/input/stroke-classification-brain-ct-dataset"
# 假设数据结构：
# DATA_DIR/
# ├── stroke/    (中风)
# └── normal/    (正常)

class CTDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        classes = {"stroke": 0, "normal": 1}  # 0=中风, 1=正常
        
        for cls_name, cls_idx in classes.items():
            cls_dir = os.path.join(data_dir, split, cls_name)
            if not os.path.exists(cls_dir):
                cls_dir = os.path.join(data_dir, cls_name)
            
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_idx)
        
        print(f"  加载 {len(self.images)} 张图片")
        print(f"  中风: {sum(1 for l in self.labels if l==0)} 张")
        print(f"  正常: {sum(1 for l in self.labels if l==1)} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# 加载数据
dataset = CTDataset(DATA_DIR, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ═══════════════════════════════════════════════
# Cell 6: CLIP零样本分类
# ═══════════════════════════════════════════════

all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="CLIP分类中"):
        images = images.to(device)
        
        # 编码图片
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 匹配文字（用类别级描述）
        similarity = (100.0 * image_features @ class_features.T)
        probs = similarity.softmax(dim=-1)
        
        preds = probs.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print(f"\n✅ 分类完成! 共 {len(all_labels)} 张图片")

# ═══════════════════════════════════════════════
# Cell 7: 评估结果
# ═══════════════════════════════════════════════

# 准确率
acc = accuracy_score(all_labels, all_preds)
print(f"\n🎯 总体准确率: {acc:.2%}")

# 分类报告
print(f"\n📊 分类报告:")
print(classification_report(all_labels, all_preds, 
      target_names=["中风(stroke)", "正常(normal)"]))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["预测中风", "预测正常"],
            yticklabels=["实际中风", "实际正常"])
plt.title(f"CLIP 脑部CT中风分类 - 混淆矩阵 (准确率: {acc:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ 混淆矩阵已保存: confusion_matrix.png")

# ═══════════════════════════════════════════════
# Cell 8: 查看分类置信度分布
# ═══════════════════════════════════════════════

stroke_probs = all_probs[all_labels == 0, 0]  # 中风图片→预测中风的概率
normal_probs = all_probs[all_labels == 1, 1]  # 正常图片→预测正常的概率

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(stroke_probs, bins=20, alpha=0.7, color='red', label='中风图片')
plt.axvline(x=0.5, color='gray', linestyle='--')
plt.xlabel('预测为中风的概率')
plt.ylabel('图片数')
plt.legend()
plt.title('中风图片的置信度分布')

plt.subplot(1, 2, 2)
plt.hist(normal_probs, bins=20, alpha=0.7, color='green', label='正常图片')
plt.axvline(x=0.5, color='gray', linestyle='--')
plt.xlabel('预测为正常的概率')
plt.ylabel('图片数')
plt.legend()
plt.title('正常图片的置信度分布')

plt.tight_layout()
plt.savefig("confidence_distribution.png")
print("✅ 置信度分布图已保存: confidence_distribution.png")

# ═══════════════════════════════════════════════
# Cell 9: 展示部分预测结果
# ═══════════════════════════════════════════════

def show_predictions(dataset, preds, labels, probs, num_samples=8):
    indices = np.random.choice(len(labels), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img_path = dataset.images[idx]
        img = Image.open(img_path)
        true_label = "中风" if labels[idx] == 0 else "正常"
        pred_label = "中风" if preds[idx] == 0 else "正常"
        prob = probs[idx][0] if preds[idx] == 0 else probs[idx][1]
        correct = "✅" if labels[idx] == preds[idx] else "❌"
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"实际: {true_label}\n预测: {pred_label} ({prob:.0%}) {correct}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("✅ 预测结果展示图已保存: sample_predictions.png")

show_predictions(dataset, all_preds, all_labels, all_probs)

print("\n🎉 CLIP零样本分类完成！")
print(f"   模型: CLIP ViT-B/32")
print(f"   数据集: 脑部CT中风分类")
print(f"   方法: 图像-文本匹配 (图文分类)")
print(f"   总体准确率: {acc:.2%}")
