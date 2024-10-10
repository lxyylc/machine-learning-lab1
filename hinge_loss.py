import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 数据集导入
from ucimlrepo import fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
print("y: ",y)
def predict(X, W, b):
    return np.sign(np.dot(X, W) + b)
# 对目标变量进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y.values)  # 确保y是一维数组

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化参数
W = np.random.randn(X_train.shape[1])
b = 0

# 学习率和迭代次数
learning_rate = 0.001
epochs = 500
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
# 训练SVM模型
for epoch in range(epochs):
    if epoch%50==0:
        y_pred = predict(X_train, W, b)
        # 计算准确率
        accuracy = accuracy_score(y_train, y_pred)
        print(f'epoch: {epoch} SVM with Hinge Loss Accuracy(training): {accuracy}')
    for i in range(len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]
        pred = np.dot(x_i, W) + b
        margin = max(0, 1 - y_i * pred)
        W += learning_rate * (y_i * x_i * margin)
        b += learning_rate * (y_i * margin)

# 预测测试集
y_pred = predict(X_test, W, b)
y_pred_train = predict(X_train, W, b)

# 计算准确率
accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)

# 计算精确率
precision_test = precision_score(y_test, y_pred)
precision_train = precision_score(y_train, y_pred_train)

# 计算召回率
recall_test = recall_score(y_test, y_pred)
recall_train = recall_score(y_train, y_pred_train)

# 计算F1值
f1_test = f1_score(y_test, y_pred)
f1_train = f1_score(y_train, y_pred_train)

# 打印训练集和测试集的全部指标
print(f'Training set metrics - Accuracy: {accuracy_train}, Precision: {precision_train}, Recall: {recall_train}, F1 Score: {f1_train}')
print(f'Testing set metrics - Accuracy: {accuracy_test}, Precision: {precision_test}, Recall: {recall_test}, F1 Score: {f1_test}')