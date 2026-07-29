"""
📈 逻辑回归 — 预测考试能不能及格
用学习时长+睡眠质量，算及格的概率
"""

import math

# 训练数据：[学习时长, 睡眠质量, 是否及格(1=及格, 0=不及格)]
data = [
    [1, 3, 0],
    [2, 4, 0],
    [3, 5, 0],
    [4, 6, 1],
    [5, 7, 1],
    [6, 8, 1],
    [8, 9, 1],
    [10, 10, 1],
]

# Sigmoid函数：把任意数字变成0~1的概率
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# 训练：找最合适的w和b
def train(data, lr=0.1, epochs=1000):
    w1, w2, b = 0.0, 0.0, 0.0  # 初始值
    n = len(data)
    
    for epoch in range(epochs):
        dw1, dw2, db = 0.0, 0.0, 0.0
        
        for hours, sleep, label in data:
            z = w1 * hours + w2 * sleep + b
            pred = sigmoid(z)
            error = pred - label
            
            # 梯度
            dw1 += error * hours
            dw2 += error * sleep
            db += error
        
        # 更新参数
        w1 -= lr * dw1 / n
        w2 -= lr * dw2 / n
        b -= lr * db / n
    
    return w1, w2, b

# 预测
def predict(w1, w2, b, hours, sleep):
    z = w1 * hours + w2 * sleep + b
    prob = sigmoid(z)
    return prob

# 跑训练
w1, w2, b = train(data)

print('📈 逻辑回归 — 考试及格预测')
print('=' * 35)
print(f'训练数据：{len(data)}条')
print()

print('📐 训练结果：')
print(f'  学习时长权重(w1) = {w1:.2f}')
print(f'  睡眠质量权重(w2) = {w2:.2f}')
print(f'  偏置(b) = {b:.2f}')
print()

print('🔮 预测新数据：')
print(f'{"学习时长":>6} {"睡眠":>4} → 及格概率')
print('-' * 30)
test_data = [(2, 4), (4, 6), (5, 7), (10, 10)]
for hours, sleep in test_data:
    prob = predict(w1, w2, b, hours, sleep)
    result = '及格✅' if prob >= 0.5 else '不及格❌'
    print(f'{hours:>6}h  {sleep:>4}分 → {prob*100:>5.1f}% {result}')
