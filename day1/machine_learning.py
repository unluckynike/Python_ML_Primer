
'''
@Project ：Python_ML_Primer 
@File    ：machine_learning.py
@Author  ：hailin
@Date    ：2022/10/15 08:31 
@Info    : 
'''
from sklearn.datasets import load_iris # 鸢尾花数据
from sklearn.model_selection import train_test_split

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    iris=load_iris()
    print("鸢尾花数据集:\n",iris) # array numpy.ndarray
    print("查看数据描述:\n",iris["DESCR"])
    print("查看特征值名字：\n",iris.feature_names)
    print("查看标签名字：\n",iris.target_names)

   # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值:\n",x_train,x_train.shape)
    return None

if __name__ == '__main__':
    datasets_demo();