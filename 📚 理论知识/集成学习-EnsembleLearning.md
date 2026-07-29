---
created: 2026-07-29
tags:
  - ML
  - AI基础
  - 集成学习
  - 监督学习
---

# 🌲 集成学习（Ensemble Learning）

## 一句话

> **三个臭皮匠，顶个诸葛亮。**
> 一个模型不准，多个模型一起投票/平均，准确率就上去了。

## 为什么有效？

单个模型容易过拟合。但多个有差异的模型一起决策，错的概率大大降低。

## 两大主流方法

### 方法①：Bagging → 随机森林

**核心：** 并行训练多个模型，取平均/投票

**两个随机：**
1. **随机选样本** — 每棵树有放回抽样，用的数据不一样
2. **随机选特征** — 每棵树只随机选部分特征来分裂

**代码3行：**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

| 优点 | 缺点 |
|:----|:----|
| 不容易过拟合 | 模型大、速度慢 |
| 能处理高维数据 | 可解释性不如单棵树 |
| 有feature_importances_ | — |

### 方法②：Boosting → XGBoost

**核心：** 串行训练，后一棵树专门纠正前一棵的错

**三种主流：**
| 算法 | 年份 | 特点 |
|:----|:---:|:----|
| AdaBoost | 1995 | 给错题加权 |
| XGBoost | 2014 | 加正则化+并行优化 🏆 |
| LightGBM | 2017 | 叶子结点生长，更快 |

### 核心参数

**随机森林：**
```python
n_estimators=100    # 树的数量
max_depth=5         # 树深度（防过拟合）
min_samples_split=2 # 节点最少样本数
```

**XGBoost：**
```python
learning_rate=0.3   # 学习率
n_estimators=100    # 树的数量
max_depth=6         # 树深度
subsample=0.8       # 每棵树用80%数据
```

### ③ Stacking（堆叠）

训练一个"裁判模型"来综合各模型的预测结果。

### ④ 硬投票 vs 软投票

| 硬投票 | 软投票 ✅ |
|:-----|:---------|
| 每个模型投一票(0/1) | 每个模型输出概率取平均 |
| 信息量少 | 信息量更多 |

### ⑤ OOB评估（Out-of-Bag）

Bagging约37%数据不会被抽到 → 天然验证集

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)
```

### 完整知识图

```
集成学习
  ├─ Bagging（并行）
  │     └─ 随机森林 ← 最常用
  ├─ Boosting（串行）
  │     ├─ AdaBoost
  │     ├─ Gradient Boosting
  │     │     ├─ XGBoost 🏆
  │     │     ├─ LightGBM
  │     │     └─ CatBoost
  └─ Stacking（分层）
        └─ 元模型整合
```

### 什么时候用哪个？

| 需求 | 选 |
|:----|:---|
| 快速出结果 | 随机森林 🌲 |
| 竞赛最高准确率 | XGBoost 🏆 |
| 看特征重要性 | 随机森林 |
| 超大数据（百万级） | XGBoost/LightGBM（GPU加速） |

## 今天的学习记录

- 2026-07-29：集成学习框架 + Bagging/随机森林 + Boosting/XGBoost + Stacking
- 前序算法：KNN、决策树、朴素贝叶斯、逻辑回归
- 下一步：监督学习还剩SVM、岭回归/Lasso
