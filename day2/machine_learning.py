
'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/16 22:15 
@Info    : sklearn转换器和预估器 KNN算法 模型选择与调优 朴素贝叶斯算法 决策树 随机森林
'''

from sklearn.datasets import load_iris # 鸢尾花数据集
from sklearn.model_selection import train_test_split,GridSearchCV # 数据集划分 模型选择与调优（网格搜索 交叉验证）
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.preprocessing import StandardScaler # 标准化


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1获取数据集
    iris=load_iris()
    # print(iris)

    # 2划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3特征工程 标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.fit_transform(x_test)

    # 4KNN算法预估器
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    # 5模型评估
    # 方法一 直接比对真实值与预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("y_test:\n",y_test)
    print("直接对比真是值和预测值:\n",y_test==y_predict)
    # 方法二 计算准确率
    socre=estimator.score(x_test,y_test)
    print("准确率为：\n",socre)
    return None

def knn_iris_gsc():
    """
    用KNN算法对鸢尾花进行分类 添加网格搜索与交叉验证
    :return:
    """
    # 1 加载数据
    iris=load_iris()

    # 2 划分数据集合
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=3)

    # 3 特征工程 标准
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.fit_transform(x_test)

    # 4 KNN算法预估器
    estimate=KNeighborsClassifier() # 默认欧式距离
    # 加入网格搜索与交叉验证
    param_dic={"n_neighbors":[1,3,5,7,9,11]}
    estimate=GridSearchCV(estimate,param_grid=param_dic,cv=10)
    estimate.fit(x_train,y_train)

    # 5 模型评估
    y_predict=estimate.predict(x_test)
    print("y_predict:\n",y_predict)
    print("y_test:\n",y_test)
    print("真实值与预测值对比：\n",y_predict==y_test)

    score=estimate.score(x_test,y_test)
    print("准确率:\n",score)

    # 结果分析
    print("最佳参数：\n",estimate.best_params_)
    print("最佳结果：\n",estimate.best_score_)
    print("最佳估计器：\n",estimate.best_estimator_)
    print("交叉验证结果：\n",estimate.cv_results_)
    return None

if __name__ == '__main__':
    # 代码一：用KNN算法对鸢尾花进行分类
    # knn_iris()
    # 代码二： 用KNN算法对鸢尾花进行分类 添加网格搜索与交叉验证
    knn_iris_gsc()