from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

print("="*50)
print("交叉验证：科学选K值")
print("="*50)
print(f"{'K值':>4} {'5折平均准确率':>15} {'每次结果':>30}")
print("-"*50)

for k in [1, 3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5折交叉验证
    scores = cross_val_score(knn, X, y, cv=5)
    mean_score = scores.mean()
    
    # 格式化5次结果
    scores_str = "  ".join([f"{s:.1%}" for s in scores])
    print(f"{k:>4} {mean_score:>15.2%}  {scores_str}")

print()
print("💡 最佳K值看平均准确率最高的那个")
print("   K=1 准确率最高，可能有轻微过拟合")
print("   实际上K=3~7都是不错的选择，选个简单稳定的")
