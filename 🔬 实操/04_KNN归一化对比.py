from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np

wine = load_wine()
X, y = wine.data, wine.target

print("="*55)
print("交叉验证（5折）对比：归一化对KNN的影响")
print("="*55)
print(f"{'K值':>4} {'不归一化(均值)':>15} {'归一化后(均值)':>15} {'提升':>10}")
print("-"*50)

for k in [1, 3, 5, 7, 9, 11, 15]:
    # 不归一化
    knn = KNeighborsClassifier(n_neighbors=k)
    scores_raw = cross_val_score(knn, X, y, cv=5)
    
    # 归一化后
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    scores_norm = cross_val_score(pipeline, X, y, cv=5)
    
    mean_raw = scores_raw.mean()
    mean_norm = scores_norm.mean()
    boost = (mean_norm - mean_raw) * 100
    
    print(f"{k:>4} {mean_raw:>15.2%} {mean_norm:>15.2%} {boost:>+9.2f}%")

print()
print("💡 结论：")
print("  - 数据集特征量级差异大时，归一化对KNN提升明显")
print("  - 归一化的本质：让所有特征公平参与距离计算")
print("  - 特征量级差异小了，归一化效果就不明显")
