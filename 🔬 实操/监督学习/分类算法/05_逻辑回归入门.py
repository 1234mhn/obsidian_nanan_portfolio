import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 画图
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', linewidth=2, label='Sigmoid函数')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='阈值=0.5')
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
plt.xlabel('输入分数')
plt.ylabel('输出概率')
plt.title('Sigmoid：把任意数字变成0~1之间的概率')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('/tmp/sigmoid.png')
print("✅ Sigmoid函数图已保存")

print("="*45)
print("看看Sigmoid把各种分数转成概率：")
print("="*45)
for score in [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]:
    prob = sigmoid(score)
    print(f"  分数 = {score:>+5.1f}  →  概率 = {prob:.3f}  →  ", end="")
    print("预测为正类 ✅" if prob >= 0.5 else "预测为负类 ❌")

# 用鸢尾花数据跑逻辑回归
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
# 逻辑回归默认处理二分类，我们用前两类
X = iris.data[:100]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*45}")
print(f"逻辑回归在鸢尾花前两类上的准确率：{acc:.2%}")
print(f"{'='*45}")
