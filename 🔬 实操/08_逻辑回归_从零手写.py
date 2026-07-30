"""
🤖 逻辑回归 · 从零手写版
2026.7.30 手打一遍建立直觉
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 1. Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 2. 前向传播：算预测概率
def predict_proba(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


# 3. 交叉熵 Loss
def compute_loss(y_hat, y):
    m = len(y)
    loss = -1 / m * np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
    return loss


# 4. 训练：梯度下降
def train(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        y_hat = predict_proba(X, w, b)

        loss = compute_loss(y_hat, y)
        loss_history.append(loss)

        dw = 1 / m * np.dot(X.T, (y_hat - y))
        db = 1 / m * np.sum(y_hat - y)

        w = w - lr * dw
        b = b - lr * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    return w, b, loss_history


# 5. 预测类别
def predict(X, w, b):
    return (predict_proba(X, w, b) >= 0.5).astype(int)


# 6. 主程序
if __name__ == "__main__":
    # 生成数据
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练
    w, b, loss_history = train(X_train, y_train, lr=0.1, epochs=1000)

    # 测试集准确率
    y_pred = predict(X_test, w, b)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n测试集准确率: {accuracy:.2%}")

    # 画 Loss 下降曲线
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch（迭代次数）")
    plt.ylabel("Loss")
    plt.title("训练过程中 Loss 的变化")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/root/.openclaw/workspace/loss_curve.png")
    print("Loss 曲线已保存为 loss_curve.png")
