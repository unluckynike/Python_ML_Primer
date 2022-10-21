'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/16 22:15 
@Info    : sklearn转换器和预估器 KNN算法 模型选择与调优 朴素贝叶斯算法 决策树 随机森林
'''

from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.model_selection import train_test_split, GridSearchCV  # 数据集划分 模型选择与调优（网格搜索 交叉验证）
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.datasets import fetch_20newsgroups  # 20新闻分类
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征抽取
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier,export_graphviz # 决策树 决策树可视化


def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1获取数据集
    iris = load_iris()
    # print(iris)

    # 2划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3特征工程 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # x_test = transfer.fit_transform(x_test)

    # 4KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5模型评估
    # 方法一 直接比对真实值与预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    print("直接对比真是值和预测值:\n", y_test == y_predict)
    # 方法二 计算准确率
    socre = estimator.score(x_test, y_test)
    print("准确率为：\n", socre)
    return None


def knn_iris_gsc():
    """
    用KNN算法对鸢尾花进行分类 添加网格搜索与交叉验证
    k取值，过小容易受到异常值定影响，过大容易受到样本不均衡定影响
    应用场景：少量数据
    :return:
    """
    # 1 加载数据
    iris = load_iris()

    # 2 划分数据集合
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=3)

    # 3 特征工程 标准
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # x_test = transfer.fit_transform(x_test)

    # 4 KNN算法预估器
    estimate = KNeighborsClassifier()  # 默认欧式距离
    # 加入网格搜索与交叉验证
    param_dic = {"n_neighbors": [1, 3, 5, 7, 9, 11]} # 多个参数放进去试
    estimate = GridSearchCV(estimate, param_grid=param_dic, cv=10)
    estimate.fit(x_train, y_train)

    # 5 模型评估
    y_predict = estimate.predict(x_test)
    print("y_predict:\n", y_predict)
    print("y_test:\n", y_test)
    print("真实值与预测值对比：\n", y_predict == y_test)

    score = estimate.score(x_test, y_test)
    print("准确率:\n", score)

    # 结果分析
    print("最佳参数：\n", estimate.best_params_)
    print("最佳结果：\n", estimate.best_score_)
    print("最佳估计器：\n", estimate.best_estimator_)
    print("交叉验证结果：\n", estimate.cv_results_)
    return None


def nb_new():
    """
    用朴素贝叶斯算法对新闻进行分类
    朴素：假定了特征与特征直接相互独立
    贝叶斯：贝叶斯公式
    拉普拉斯平滑系数
    应用场景：文本分类
    :return:
    """
    # 1获取数据
    new_data = fetch_20newsgroups(subset="all")
    # 2划分数据集
    x_train, x_test, y_train, y_test = train_test_split(new_data.data, new_data.target)
    # 3特征工程： 文本特征抽取 tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train) #  Fit to data, then transform it.
    x_test = transfer.transform(x_test) # Transform documents to document-term matrix.

    # 4朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train) # fit 方法进行训练 Fit Naive Bayes classifier according to X, y
    # 5模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率:\n", score)
    return None

def decision_iris():
    """
    用决策树对鸢尾花进行分类
    找到最高效定决策顺序-信息增益
    信息增益=信息熵-条件熵
    可视化，可解释能力强
    :return:
    """
    iris=load_iris()
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=6)

    # 决策树模预估器
    estimator=DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("y_test:\n",y_test)
    print("直接对比：\n",y_predict==y_test)

    socre=estimator.score(x_test,y_test)
    print("准确率：\n",socre)

    # 决策树可视化
    export_graphviz(estimator,out_file="iris_tree.dot",feature_names=iris.feature_names)
    return None

if __name__ == '__main__':
    # 代码一：用KNN算法对鸢尾花进行分类
    # knn_iris()
    # 代码二：用KNN算法对鸢尾花进行分类 添加网格搜索与交叉验证
    #  knn_iris_gsc()
    # 代码三：用朴素贝叶斯算法对新闻进行分类
    #  nb_new()
    # 代码四： 用决策树对鸢尾花进行分类
    decision_iris()
