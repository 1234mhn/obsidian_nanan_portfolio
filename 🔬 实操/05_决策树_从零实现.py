"""
🌲 决策树 — 从零实现
《机器学习实战》第3章
不调sklearn，自己建树自己预测
"""

import math
from collections import Counter

# ═══════════════════════════════════════
# 1. 熵 + 信息增益
# ═══════════════════════════════════════

def entropy(data):
    """计算数据集的熵"""
    total = len(data)
    if total == 0:
        return 0
    counter = Counter(data)
    e = 0.0
    for label in counter:
        p = counter[label] / total
        e -= p * math.log2(p)
    return e

def info_gain(data, feature_idx):
    """
    计算按某个特征划分的信息增益
    data: 二维数组，每行=[特征1, 特征2, ..., 标签]
    feature_idx: 用第几列特征来分
    """
    labels = [row[-1] for row in data]
    total_ent = entropy(labels)
    
    # 按特征值分组
    groups = {}
    for row in data:
        val = row[feature_idx]
        if val not in groups:
            groups[val] = []
        groups[val].append(row[-1])  # 只存标签
    
    # 加权平均熵
    weighted_ent = 0.0
    for val, group_labels in groups.items():
        w = len(group_labels) / len(data)
        weighted_ent += w * entropy(group_labels)
    
    return total_ent - weighted_ent

# ═══════════════════════════════════════
# 2. 建树
# ═══════════════════════════════════════

def majority_label(labels):
    """投票：选出现次数最多的标签"""
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

def build_tree(data, features, depth=0):
    """
    递归建树
    data: [[f1, f2, ..., label], ...]
    features: 特征名列表 ['天气', '风力', ...]
    """
    labels = [row[-1] for row in data]
    
    # 终止条件1：全是同一类 → 返回该类
    if len(set(labels)) == 1:
        return labels[0]
    
    # 终止条件2：没有特征可分了 → 返回多数票
    if len(data[0]) == 1:  # 只剩标签列
        return majority_label(labels)
    
    # 选信息增益最大的特征
    best_gain = -1
    best_idx = -1
    for i in range(len(data[0]) - 1):  # 不包含最后一列（标签）
        gain = info_gain(data, i)
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    
    best_feature = features[best_idx]
    tree = {best_feature: {}}  # 树的根节点
    
    # 按最佳特征的值分组
    groups = {}
    for row in data:
        val = row[best_idx]
        if val not in groups:
            groups[val] = []
        # 去掉已使用的特征列
        new_row = row[:best_idx] + row[best_idx+1:]
        groups[val].append(new_row)
    
    # 递归建子树
    new_features = features[:best_idx] + features[best_idx+1:]
    for val, sub_data in groups.items():
        tree[best_feature][val] = build_tree(sub_data, new_features, depth+1)
    
    return tree

# ═══════════════════════════════════════
# 3. 预测
# ═══════════════════════════════════════

def predict(tree, sample, features):
    """用训练好的树预测一条新数据"""
    if not isinstance(tree, dict):
        return tree  # 叶子节点
    
    feature_name = list(tree.keys())[0]
    feature_idx = features.index(feature_name)
    feature_val = sample[feature_idx]
    
    branch = tree[feature_name]
    if feature_val not in branch:
        # 没见过的值，返回该子树的多数票
        return None
    
    return predict(branch[feature_val], sample, features)

# ═══════════════════════════════════════
# 4. 打印树
# ═══════════════════════════════════════

def print_tree(tree, indent=''):
    """可视化树结构"""
    if not isinstance(tree, dict):
        print(indent + '→', tree)
        return
    
    feature = list(tree.keys())[0]
    print(indent + feature + '?')
    
    for val, subtree in tree[feature].items():
        print(indent + ' ├─', val, end=' ')
        if isinstance(subtree, dict):
            print()
            print_tree(subtree, indent + ' │  ')
        else:
            print('→', subtree)


# ═══════════════════════════════════════
# 5. 测试：判断要不要去跑步
# ═══════════════════════════════════════

if __name__ == '__main__':
    # 训练数据：[天气, 风力, 湿度, 标签]
    train_data = [
        ['好', '小', '低', '去'],
        ['好', '小', '高', '去'],
        ['好', '大', '低', '不去'],
        ['好', '大', '高', '不去'],
        ['差', '小', '低', '去'],
        ['差', '小', '高', '不去'],
        ['差', '大', '低', '不去'],
        ['差', '大', '高', '不去'],
    ]
    features = ['天气', '风力', '湿度']
    
    print('🌲 决策树训练')
    print('=' * 35)
    print(f'训练数据：{len(train_data)}条')
    print(f'特征：{features}')
    print()
    
    # 建树
    tree = build_tree(train_data, features)
    
    print('📐 生成的决策树：')
    print_tree(tree)
    print()
    
    # 预测
    test_data = [
        ['好', '小', '低'],
        ['好', '小', '高'],
        ['好', '大', '低'],
        ['差', '小', '低'],
        ['差', '大', '高'],
    ]
    
    print('🔮 预测结果：')
    print(f'{"天气":>3} {"风力":>3} {"湿度":>3} → 预测')
    print('-' * 20)
    for sample in test_data:
        pred = predict(tree, sample, features)
        print(f'{sample[0]:>3} {sample[1]:>3} {sample[2]:>3} → {pred}')
