from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

print("="*50)
print("🌸 逻辑回归做多分类（鸢尾花3类）")
print("="*50)
# 新版sklearn默认自动用multinomial
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)

print(f"\n测试集准确率：{acc:.2%}")
print(f"\n详细报告：")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 看概率
print(f"\n{'='*50}")
print("🔍 每类预测的概率：")
print("="*50)
probs = lr.predict_proba(X_test_s[:5])
actual = y_test[:5]
pred = y_pred[:5]

for i in range(5):
    a = iris.target_names[actual[i]]
    p = iris.target_names[pred[i]]
    ok = "✅" if actual[i] == pred[i] else "❌"
    print(f"  样本{i+1}: 实际={a:10s} 预测={p:10s}  ", end="")
    for j in range(3):
        print(f"{iris.target_names[j]}:{probs[i][j]:.1%} ", end="")
    print(ok)
print("\n概率加起来都是1！这就是softmax干的事")
