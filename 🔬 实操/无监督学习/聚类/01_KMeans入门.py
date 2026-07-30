"""
🎯 K-Means 聚类 · 入门演示
2026.7.30
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 造3堆数据
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# K-Means — 就这一行
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print(f"数据：{len(X)}个点，自动分成3堆")
print(f"每堆数量：{[sum(labels==i) for i in range(3)]}")

# 画图
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster centers')
plt.title('K-Means Clustering (K=3)')
plt.legend()
plt.savefig('kmeans_demo.png')
