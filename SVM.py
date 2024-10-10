from ucimlrepo import fetch_ucirepo
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 导入数据集
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# 数据（作为pandas数据框）
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性核的参数范围
param_grid_linear = {
    'C': [0.1, 1, 10, 100]
}

# 高斯核的参数范围
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

# 创建SVM分类器
svm_linear = SVC(kernel='linear', probability=True)  # 设置probability=True以启用概率估计
svm_rbf = SVC(kernel='rbf', probability=True)  # 设置probability=True以启用概率估计

# 创建网格搜索对象 - 线性核
grid_search_linear = GridSearchCV(estimator=svm_linear, param_grid=param_grid_linear, cv=5, scoring='accuracy')
grid_search_linear.fit(X_train_scaled, y_train)

# 打印线性核最佳参数
print("线性核最佳参数：", grid_search_linear.best_params_)

# 使用最佳参数创建最优模型 - 线性核
best_svm_linear = grid_search_linear.best_estimator_

# 在训练集和测试集上进行预测 - 线性核
y_pred_train_linear = best_svm_linear.predict(X_train_scaled)
y_pred_test_linear = best_svm_linear.predict(X_test_scaled)

# 计算并打印训练集和测试集的分类报告 - 线性核
print("\n线性核最优模型的训练集分类报告：")
print(classification_report(y_train, y_pred_train_linear))

print("\n线性核最优模型的测试集分类报告：")
print(classification_report(y_test, y_pred_test_linear))

# 计算并打印训练集和测试集的评价指标 - 线性核
accuracy_train_linear = accuracy_score(y_train, y_pred_train_linear)
precision_train_linear = precision_score(y_train, y_pred_train_linear, pos_label='M')
recall_train_linear = recall_score(y_train, y_pred_train_linear, pos_label='M')
f1_train_linear = f1_score(y_train, y_pred_train_linear, pos_label='M')

accuracy_test_linear = accuracy_score(y_test, y_pred_test_linear)
precision_test_linear = precision_score(y_test, y_pred_test_linear, pos_label='M')
recall_test_linear = recall_score(y_test, y_pred_test_linear, pos_label='M')
f1_test_linear = f1_score(y_test, y_pred_test_linear, pos_label='M')

print(f"线性核最优模型的训练集准确率：{accuracy_train_linear}")
print(f"线性核最优模型的训练集精确率：{precision_train_linear}")
print(f"线性核最优模型的训练集召回率：{recall_train_linear}")
print(f"线性核最优模型的训练集F1值：{f1_train_linear}")

print(f"线性核最优模型的测试集准确率：{accuracy_test_linear}")
print(f"线性核最优模型的测试集精确率：{precision_test_linear}")
print(f"线性核最优模型的测试集召回率：{recall_test_linear}")
print(f"线性核最优模型的测试集F1值：{f1_test_linear}")

# 创建网格搜索对象 - 高斯核
grid_search_rbf = GridSearchCV(estimator=svm_rbf, param_grid=param_grid_rbf, cv=5, scoring='accuracy')
grid_search_rbf.fit(X_train_scaled, y_train)

# 打印高斯核最佳参数
print("高斯核最佳参数：", grid_search_rbf.best_params_)

# 使用最佳参数创建最优模型 - 高斯核
best_svm_rbf = grid_search_rbf.best_estimator_

# 在训练集和测试集上进行预测 - 高斯核
y_pred_train_rbf = best_svm_rbf.predict(X_train_scaled)
y_pred_test_rbf = best_svm_rbf.predict(X_test_scaled)

# 计算并打印训练集和测试集的分类报告 - 高斯核
print("\n高斯核最优模型的训练集分类报告：")
print(classification_report(y_train, y_pred_train_rbf))

print("\n高斯核最优模型的测试集分类报告：")
print(classification_report(y_test, y_pred_test_rbf))

# 计算并打印训练集和测试集的评价指标 - 高斯核
accuracy_train_rbf = accuracy_score(y_train, y_pred_train_rbf)
precision_train_rbf = precision_score(y_train, y_pred_train_rbf, pos_label='M')
recall_train_rbf = recall_score(y_train, y_pred_train_rbf, pos_label='M')
f1_train_rbf = f1_score(y_train, y_pred_train_rbf, pos_label='M')

accuracy_test_rbf = accuracy_score(y_test, y_pred_test_rbf)
precision_test_rbf = precision_score(y_test, y_pred_test_rbf, pos_label='M')
recall_test_rbf = recall_score(y_test, y_pred_test_rbf, pos_label='M')
f1_test_rbf = f1_score(y_test, y_pred_test_rbf, pos_label='M')

print(f"高斯核最优模型的训练集准确率：{accuracy_train_rbf}")
print(f"高斯核最优模型的训练集精确率：{precision_train_rbf}")
print(f"高斯核最优模型的训练集召回率：{recall_train_rbf}")
print(f"高斯核最优模型的训练集F1值：{f1_train_rbf}")

print(f"高斯核最优模型的测试集准确率：{accuracy_test_rbf}")
print(f"高斯核最优模型的测试集精确率：{precision_test_rbf}")
print(f"高斯核最优模型的测试集召回率：{recall_test_rbf}")
print(f"高斯核最优模型的测试集F1值：{f1_test_rbf}")