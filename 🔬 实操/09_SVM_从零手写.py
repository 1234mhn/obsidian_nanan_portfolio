"""
🤖 SVM（支持向量机）· 从零手写版
2026.7.30 手打一遍：Hinge Loss + 梯度下降 + 最大间隔
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 1. 训练：用梯度下降实现 SVM（Hinge Loss）
def train(X, y, lr=0.01, epochs=1000, C=1.0):
    # 把标签从 0/1 转成 -1/+1
    y = np.where(y == 0, -1, 1)

    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        loss = 0
        for i in range(m):
            # 判断这个点是否在间隔外且分对了
            condition = y[i] * (np.dot(X[i], w) + b) >= 1

            if condition:
                dw = w
                db = 0
            else:
                dw = w - C * y[i] * X[i]
                db = -C * y[i]
                loss += 1 - y[i] * (np.dot(X[i], w) + b)

        w = w - lr * dw
        b = b - lr * db

        total_loss = 0.5 * np.dot(w, w) + C * loss
        loss_history.append(total_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f}")

    return w, b, loss_history


# 2. 预测
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


# 3. 主程序
if __name__ == "__main__":
    X, y = make_classification(n_samples=300, n_features=2,
                               n_redundant=0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    w, b, loss_history = train(X_train, y_train, lr=0.01, epochs=1000, C=1.0)

    y_pred = predict(X_test, w, b)
    y_test_converted = np.where(y_test == 0, -1, 1)
    accuracy = np.mean(y_pred == y_test_converted)
    print(f"\n测试集准确率: {accuracy:.2%}")
    print(f"学习到的 w: {w}")
    print(f"学习到的 b: {b:.4f}")
