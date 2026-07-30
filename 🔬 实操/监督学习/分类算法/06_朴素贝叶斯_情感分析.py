"""
📊 朴素贝叶斯 — 情感分析
《机器学习实战》第4章
判断一句话是正面还是负面评价
"""

def train(data):
    """
    data: [([词1,词2,...], 标签), ...]
    返回: P(类别), P(词|类别)
    """
    # 统计
    vocab = set()          # 所有不重复的词
    class_count = {}       # 每个类别出现了几次
    word_count = {}        # 每个类别下每个词出现了几次
    
    for words, label in data:
        class_count[label] = class_count.get(label, 0) + 1
        if label not in word_count:
            word_count[label] = {}
        for w in words:
            vocab.add(w)
            word_count[label][w] = word_count[label].get(w, 0) + 1
    
    total = len(data)
    class_prob = {}         # P(类别)
    word_prob = {}          # P(词|类别)
    
    for label in class_count:
        class_prob[label] = class_count[label] / total
        
        word_prob[label] = {}
        total_words = sum(word_count[label].values())
        vocab_size = len(vocab)
        
        for w in vocab:
            # 拉普拉斯平滑：+1 防止概率为0
            count = word_count[label].get(w, 0) + 1
            word_prob[label][w] = count / (total_words + vocab_size)
    
    return vocab, class_prob, word_prob


def predict(vocab, class_prob, word_prob, words):
    """
    预测一句话的情感
    """
    scores = {}
    for label in class_prob:
        score = class_prob[label]  # 先乘 P(类别)
        for w in words:
            if w in vocab:
                score *= word_prob[label].get(w, 1)
        scores[label] = score
    
    # 归一化成百分比
    total = sum(scores.values())
    for label in scores:
        scores[label] = scores[label] / total * 100
    
    return scores


# ═══════════════════════════════════
# 测试：情感分析
# ═══════════════════════════════════

train_data = [
    (['好吃', '喜欢', '推荐'], '正面'),
    (['好吃', '便宜', '满意'], '正面'),
    (['难吃', '贵', '失望'], '负面'),
    (['难吃', '服务差', '不推荐'], '负面'),
    (('好吃', '服务好', '还会来'), '正面'),
]

vocab, class_prob, word_prob = train(train_data)

print('📊 朴素贝叶斯 — 情感分析')
print('=' * 35)
print(f'训练数据：{len(train_data)}条')
print(f'词汇表：{sorted(vocab)}')
print()

print('📐 训练结果：')
for label in class_prob:
    print(f'  P({label}) = {class_prob[label]:.2f}')
print()

# 预测
test_data = [
    ['好吃', '便宜'],
    ['难吃', '服务差'],
    ['好吃', '服务差'],
    ['推荐', '满意', '还会来'],
]

print('🔮 预测结果：')
print(f'{"评价":>15} → 正面%  负面%')
print('-' * 35)
for words in test_data:
    scores = predict(vocab, class_prob, word_prob, words)
    print(f'{str(words):>15} → {scores["正面"]:>5.1f}% {scores["负面"]:>5.1f}%')
