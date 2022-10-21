'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/21 20:06 
@Info    : 回归和类聚 线性回归 岭回归 分类算法 逻辑回归 模型加载玉保存
'''

from sklearn.datasets import load_boston  # 波士顿房价数据集
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.linear_model import LinearRegression, SGDRegressor  # 线性回归 随机梯度下降


def line1():
    """
     正规方程的方法 预测波士顿房价
    :return:
    """
    # 1 获取数据集
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)  # 特征数量等于权重系数数量
    # 2 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3 特征工程 无量纲化 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4 预估器流程 fit -- > 模型 codef_intercept
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 5 得到模型
    print("正规方程权重系数为：\n", estimator.coef_)
    print("正规方程偏置为:\n", estimator.intercept_)
    # 6 模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程的均方误差:\n", error)

    return None


def line2():
    """
    梯度下降的方法 预测波士顿房价
    :return:
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = SGDRegressor(learning_rate="constant", eta0=0.005, max_iter=100000)  # 梯度下降 调参
    estimator.fit(x_train, y_train)
    print("梯度下降权重系数：\n", estimator.coef_)
    print("梯度下降偏置：\n", estimator.intercept_)
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降的均方误差:\n", error)
    return None


if __name__ == '__main__':
    # 代码一：正规方程的方法 预测波士顿房价
    line1()
    # 代码二：梯度下降的方法 预测波士顿房价
    line2()
