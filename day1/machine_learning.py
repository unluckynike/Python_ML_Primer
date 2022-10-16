'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/15 08:31 
@Info    :  特征抽取 特征预处理 特征降维 主成分分析
'''
import jieba
from sklearn.datasets import load_iris  # 鸢尾花数据
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer  # 特征提取
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer  # 文本特征提取
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 归一化 标准化
from sklearn.feature_selection import VarianceThreshold # 降维
from sklearn.decomposition import PCA # PCA降维
import pandas as pd
from scipy.stats import pearsonr # 皮尔逊相关系数

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

def cut_word(text):
    """
   进行中文分词："我爱北京天安门" --> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    text=" ".join(list(jieba.cut(text)))
    # print(type(text)) # str字符串类型
    return text

def count_chinese_demo_two():
    """
    中文文本特征提取自动分词
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new=[]
    # 将中文文本进行分词
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    transfer=CountVectorizer(stop_words=["一种","今天"])# stop_words 停用词
    data_final=transfer.fit_transform(data_new)
    print("data_final:\n",data_final)
    print("特征名称：\n",transfer.get_feature_names_out())
    return None

def tfidf():
    """
    用TFIDF的方法进行文本特征抽取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    transfer=TfidfVectorizer()
    data_final=transfer.fit_transform(data_new)
    print("data_final:\n",data_final.toarray())# 数值大比较有分类意义
    print("特征名称：\n",transfer.get_feature_names_out())
    return None

def minmax_demo():
    """
    归一化
    :return:
    """
    # 1 获取数据
    # 2 实例化转换器
    # 3 调用fit_transform
    data = pd.read_csv("dating.txt") # 约会数据 milage,Liters,Consumtime,target
    data=data.iloc[:,:3] # 取前三列
    # print("data:\n",data.head(10))
    # transfer=MinMaxScaler() # 默认是转为0-1
    transfer=MinMaxScaler(feature_range=[2,3])# 手动设置为2-3
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def stand_demo():
    """
    标准化
    :return:
    """
    data=pd.read_csv("dating.txt")
    # print("data:\n",data.head())
    data=data.iloc[:,:3] # 第一个冒号为索引index，第二个冒号（：3）表示取前四列
    # print("data:\n", data.head())
    transfer=StandardScaler()
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def variance_demo():
    """
    低方差特征过滤
    :return:
    """
    # 1 获取数据
    # 2 实例化一个转换器类
    # 3 调用调用fit_transform
    data=pd.read_csv("factor_returns.csv")
    # index,pe_ratio,pb_ratio,market_cap,return_on_asset_net_profit,du_return_on_equity,ev,earnings_per_share,revenue,total_expense,date,return
    print("data:\n", data.head())
    data=data.iloc[:,1:-2]# 行全要 ， 列从第一列到 倒数第二列
    # print("data:\n", data.head())
    transfer=VarianceThreshold() # 可以设置阈值过滤低方差特征
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    print("shape:",data_new.shape) # 原来是9列现在还是9列，说明没有方差为0的
    # 计算某两个变量之间的相关系数 皮尔森相关系数
    """
    pearsonr
    Returns
    -------
    r : float
        Pearson's correlation coefficient.
    p-value : float
        Two-tailed p-value.
    """
    r=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("pe_ratio,pb_ratio 皮尔森相关系数：",r)
    r2=pearsonr(data["revenue"],data["total_expense"])
    print("revenue,total_expense 皮尔森相关系数：", r2)# (0.9958450413136078, 0.0)
    return None

def pac_demo():
    """
    PAC Principal components analysis 降维 主成分分析
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    transfer=PCA(n_components=2) # 四个特征降为两个特征
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

if __name__ == '__main__':
    # 代码一：sklearn数据集使用
    """ 特征抽取 """
    # datasets_demo();
    # 代码二：字典特征提取
    # dict_demo()
    # 代码三：文本特征提取
    # count_demo()
    # 代码四：中文文本特征提取
    # count_chinese_demo()
    # print("cut word:\n",cut_word("我们活在浩瀚的宇宙中，漫天飘洒的宇宙尘埃和星河光尘，我们是比这些还要渺小的存在"))
    # 代码五：中文文本特征提取 自动分词
    # count_chinese_demo_two()
    # 代码六： 用TFIDF的方法进行文本特征抽取
    # tfidf()
    """ 特征预处理 """
    # 代码七：归一化
    # minmax_demo()
    # 代码八：标准化
    # stand_demo()
    """ 特征降维 """
    # 代码九：低方差特征过滤
    # variance_demo()
    # 代码十：PAC降维
    pac_demo()