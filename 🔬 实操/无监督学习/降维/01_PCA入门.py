"""
📉 PCA（主成分分析）· 入门演示
2026.7.30
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 鸢尾花数据：4个特征
iris = load_iris()
X = iris.data       # 4个特征：花萼长、花萼宽、花瓣长、花瓣宽
y = iris.target     # 3种花的标签

print(f"原始数据：{X.shape[1]}个特征，{X.shape[0]}条数据")

# PCA降到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"PCA降维后：{X_pca.shape[1]}个特征")
print(f"第一主成分保留信息：{pca.explained_variance_ratio_[0]:.1%}")
print(f"第二主成分保留信息：{pca.explained_variance_ratio_[1]:.1%}")
print(f"总共保留了：{sum(pca.explained_variance_ratio_):.1%} 的原始信息")

# 画图对比
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Original (first 2 features)')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA (retained {sum(pca.explained_variance_ratio_):.0%} info)')

plt.tight_layout()
plt.savefig('pca_demo.png')
print("\n✅ 演示完成！4维→2维，保留98%信息")
