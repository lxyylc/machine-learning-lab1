import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# 加载数据
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# 将标签数值化
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 定义多项式拟合函数
def polynomial_fit(X_train_scaled, y_train, degree):
    # 使用numpy.polyfit进行多项式拟合
    coefficients = np.polyfit(X_train_scaled[:, 0], y_train, degree)
    polynomial = np.poly1d(coefficients)

    # 绘制训练数据
    plt.scatter(X_train_scaled[:, 0], y_train, color='red', label='Training data')

    # 绘制测试数据
    plt.scatter(X_test_scaled[:, 0], y_test, color='blue', label='Test data')

    # 绘制拟合曲线
    x_range = np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 100)
    plt.plot(x_range, polynomial(x_range), color='green', label=f'Polynomial degree {degree}')

    # 评估模型
    y_pred = polynomial(X_test_scaled[:, 0])
    y_pred_class = np.round(y_pred)  # 将连续值转换为类别标签
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Degree {degree}: Accuracy = {accuracy:.4f}")

    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Label')
    plt.title(f'Polynomial Fit with Degree {degree}')
    plt.ylim(-1,2)
    plt.legend()
    plt.show()


# 尝试不同的多项式度数
degrees = [1, 2,3,5,10,20,25]  # 线性、二次、三次
for degree in degrees:
    polynomial_fit(X_train_scaled, y_train, degree)