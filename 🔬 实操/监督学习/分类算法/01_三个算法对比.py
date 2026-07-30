import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

# 韩语视频场景数据
data = np.array([
    [100, 8, 2, 3000],
    [200, 15, 5, 8000],
    [150, 10, 3, 5000],
    [50, 2, 0, 500],
    [300, 25, 10, 15000],
    [80, 5, 1, 2000],
    [250, 20, 8, 12000],
])

X = data[:, :3]  # 点赞、评论、分享
y = data[:, 3]   # 播放量

print("="*60)
print("韩语视频播放量预测 - 三个算法对比")
print("="*60)

# 线性回归
lr = LinearRegression()
lr.fit(X, y)

# KNN回归
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X, y)

# 逻辑回归
y_binary = (y >= 5000).astype(int)
logreg = LogisticRegression()
logreg.fit(X, y_binary)

# 新视频预测
new_videos = np.array([
    [180, 12, 4],
    [400, 30, 15],
    [30, 1, 0],
])

print(f"\n  {'点赞':>4} {'评论':>4} {'分享':>4}  {'线性回归':>8} {'KNN回归':>8} {'逻辑回归':>12}")
print("  " + "-"*45)
for vid in new_videos:
    lr_pred = lr.predict([vid])[0]
    knn_pred = knn.predict([vid])[0]
    lr_class = "能破万" if logreg.predict([vid])[0] == 1 else "不能破万"
    lr_prob = logreg.predict_proba([vid])[0][1]
    print(f"  {vid[0]:>4} {vid[1]:>4} {vid[2]:>4}  {lr_pred:>8.0f} {knn_pred:>8.0f} {lr_class:>12}({lr_prob:.0%})")

# 训练集效果
print(f"\n训练集效果对比：")
lr_train = lr.predict(X)
knn_train = knn.predict(X)

for i in range(len(X)):
    print(f"  实际{y[i]:>6} 线性{lr_train[i]:>8.0f}  KNN{knn_train[i]:>8.0f}")

print()
print("三个算法各有各的用处：")
print("  - 想知道具体数字 → 线性回归")
print("  - 想看相似视频 → KNN回归")
print("  - 想做判断决策 → 逻辑回归")
