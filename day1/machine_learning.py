'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/15 08:31 
@Info    : 
'''
from sklearn.datasets import load_iris  # 鸢尾花数据
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer  # 特征提取
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征提取


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    iris = load_iris()
    print("鸢尾花数据集:\n", iris)  # array numpy.ndarray
    print("查看数据描述:\n", iris["DESCR"])
    print("查看特征值名字：\n", iris.feature_names)
    print("查看标签名字：\n", iris.target_names)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n", x_train, x_train.shape)
    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    # 1.实例化一个转换器类
    transfer = DictVectorizer(sparse=False)  # sparse 稀疏 将非零值按位置表示出来 节省内存 提高加载效率
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("feature_names:\n", transfer.get_feature_names_out())
    return None


def count_demo():
    """
    文本特征抽取
    :return:
    """
    data = ["life is short,i like like python",
            "life is too long,i dislike python",
            "i like java,java is very nice"]
    # 1.实例化一个转换器
    transfer = CountVectorizer()  # 统计每个样本特征词出现的个数
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)  # data_new每一行数字代表特征值出现的次数
    print("date_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

def count_chinese_demo():
    """
    中文文本特征提问
    :return:
    """
    data=["我 爱 北京 天安门", "天安门 上 太阳 升"]
    transfer=CountVectorizer()
    data_new=transfer.fit_transform(data)
    print("type(data_new): ",type(data_new))
    print("data_new:\n",data_new)
    print("data_new:\n",data_new.toarray())
    print("特征名称：\n",transfer.get_feature_names_out())

    return None

if __name__ == '__main__':
    # 代码一：sklearn数据集使用
    # datasets_demo();
    # 代码二：字典特征提取
    # dict_demo()
    # 代码三：文本特征提取
    # count_demo()
    # 代码四：中文文本特征提取
    count_chinese_demo()