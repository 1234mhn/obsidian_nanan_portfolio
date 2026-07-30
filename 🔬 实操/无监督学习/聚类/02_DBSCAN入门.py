"""
🎯 DBSCAN — 比K-Means更强的聚类
2026.7.30
对比：月亮形状数据，K-Means vs DBSCAN
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons

# 造月亮形状数据（K-Means搞不定那种）
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

print(f"数据：{len(X)}个点，月亮形状")

# K-Means 试试
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN 试试
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

n_noise = sum(dbscan_labels == -1)
print(f"K-Means分出来：{len(set(kmeans_labels))}堆")
print(f"DBSCAN分出来：{len(set(dbscan_labels)) - (1 if n_noise > 0 else 0)}堆 + 离群点{n_noise}个")

# 画图对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means (wrong shape)')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50)
plt.title('DBSCAN (perfect!)')
plt.tight_layout()
plt.savefig('dbscan_demo.png')
print("\n✅ K-Means乱分，DBSCAN完美搞定")
