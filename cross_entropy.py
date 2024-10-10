import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 数据集导入
from ucimlrepo import fetch_ucirepo
import warnings

# 禁用所有运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# 对目标变量进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 将 y 转换为一维数组

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2)

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

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 训练逻辑回归模型
for epoch in range(epochs):
    for i in range(len(X_train)):
        x_i = X_train[i]
        y_i = y_train[i]
        pred = sigmoid(np.dot(x_i, W) + b)
        loss_gradient_w = (pred - y_i) * x_i
        loss_gradient_b = (pred - y_i)
        W -= learning_rate * loss_gradient_w
        b -= learning_rate * loss_gradient_b
    if epoch % 10 == 0 or epoch == epochs - 1:
        y_pred = sigmoid(np.dot(X_train, W) + b)
        loss = cross_entropy_loss(y_train, y_pred)
        accuracy = accuracy_score(y_train, y_pred >= 0.5)
        print(f'Epoch {epoch}:Accuracy(training): {accuracy:.4f}')
# 最终预测
y_pred_final_train = sigmoid(np.dot(X_train, W) + b) >= 0.5
y_pred_final_test = sigmoid(np.dot(X_test, W) + b) >= 0.5

# 计算训练集和测试集的指标
accuracy_train = accuracy_score(y_train, y_pred_final_train)
precision_train = precision_score(y_train, y_pred_final_train)
recall_train = recall_score(y_train, y_pred_final_train)
f1_train = f1_score(y_train, y_pred_final_train)

accuracy_test = accuracy_score(y_test, y_pred_final_test)
precision_test = precision_score(y_test, y_pred_final_test)
recall_test = recall_score(y_test, y_pred_final_test)
f1_test = f1_score(y_test, y_pred_final_test)

# 打印训练集和测试集的全部指标
print(f'Training set metrics - Accuracy: {accuracy_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1 Score: {f1_train:.4f}')
print(f'Testing set metrics - Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1 Score: {f1_test:.4f}')