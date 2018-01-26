import argparse
import csv
import numpy as np
import argparse
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor    


class FastResearchData(object):
    """
        加载数据 转化格式
    """
    def __init__(self):
        self.dataset = DataFrame([])
        self.model = []

    def loadFromDataFrame(self, df):
        self.dataset = df
    
    def loadFromCSV_CSV(self, file_name, action="rb"):
        """
            加载csv的不同方式:使用csv包
        """
        head_list = []
        data_list = []
        with open(file_name, action)as File:
            lines = csv.reader(File)
            for i, line in enumerate(lines):  # enumerate
                if i == 0:
                    head_list = line
                else:
                    data_list.append(line)
        data = DataFrame(data_list)
        data.columns = head_list
        self.dataset = data

    def loadFromCSV_Numpy(self, file_name):
        """
            加载csv的不同方式:使用numpy
        """
        tmp = np.loadtxt(file_name, dtype=np.str, delimiter=",")
        data = DataFrame(tmp[1:, 0:])
        data.columns = tmp[0, 0:]
        self.dataset = data

    def loadFromCSV_Pandas(self, file_name):
        """
            加载csv的不同方式:使用pandas
        """
        data = pd.read_csv(file_name)
        self.dataset = data

    def getDataFrame(self):
        """
            得到dataframe方便进行后面的指标选择和计算
        """
        return self.dataset


class IndicatorGallexy(object):
    def __init__(self, df):
        self.dataset = df
        self.indicator = DataFrame([])

    def calLogPrice(self):
        """
            log(close) - log(open)
        """
        open_price = self.dataset['open']
        close_price = self.dataset['close']
        closelessopen = []
        for x, y in zip(open_price, close_price):
            tmp = math.log(y) - math.log(x)
            closelessopen.append(tmp)
        indictor_tmp = DataFrame(closelessopen)
        indictor_tmp.columns = ["log_price"]
        self.indicator = indictor_tmp

    """
        MACD:指数平滑移动平均线，是从双指数移动平均线发展而来的，由快的指数平均线（EMA12）减去慢的指数移动平均线（EMA26）
        得到快线DIF，再用2*（快线DIF-DIF的九日加权移动均线DEA）得到MACD柱
    """

    def calEMA(self, shortNumber, longNumber):
        """
            计算移动平均值，快速移动平均线为12日，慢速移动平均线为26日
            快速：EMA[i] = EMA[i-1] * (short - 1)/(short + 1) + close * 2 / (short + 1) 
            慢速：EMA[i] = EMA[i-1] * (long - 1)/(long + 1) + close * 2 / (long + 1) 
        """
        ema_short = [self.dataset['close'][0]] * len(self.dataset)
        ema_long = [self.dataset['close'][0]] * len(self.dataset)
        for i in range(1, len(self.dataset)):
            ema_short[i] = ema_short[i-1] * (shortNumber-1)/(shortNumber+1) + self.dataset['close'][i] * 2/(shortNumber+1)
            ema_long[i] = ema_long[i-1] * (longNumber-1)/(longNumber+1) + self.dataset['close'][i] * 2/(longNumber+1)
        ema_short = DataFrame(ema_short)
        ema_short.columns = ["ema_12"]  
        ema_long = DataFrame(ema_long)
        ema_long.columns = ["ema_26"]
        self.indicator = self.indicator.join(ema_short)
        self.indicator = self.indicator.join(ema_long)

    def calDIF(self, shortNumber=12, longNumber=26):
        """
            DIF为离差值，涨势中，离差值会变得越来越大，跌势中，离差值会变得越来越小
            DIF = EMA(short) - EMA(long)
        """
        self.calEMA(shortNumber, longNumber)
        ema_data = self.indicator["ema_12"] - self.indicator["ema_26"]
        dif = DataFrame(ema_data)
        dif.columns = ["dif"]
        self.indicator = self.indicator.join(dif)  
        # join 为索引连接， 如果其中一个为空，则不能进行连接

    def calDEA(self, n=9):
        """
            计算DEA差离平均值
            DEA[i] = DEA[i-1] * (n-1) / (n+1) + DIF[i] * 2 / (n+1)
            其中n为多少日
        """
        dea = [self.indicator['dif'][0]] * len(self.dataset)
        for i in range(1, len(self.dataset)):
            dea[i] = dea[i-1] * (n-1)/(n+1) + self.indicator['dif'][i] * 2/(n+1)
        dea = DataFrame(dea)
        dea.columns = ["dea"]
        self.indicator = self.indicator.join(dea)

    def calMACD(self):
        """
            计算MACD指数平滑移动平均线
            MACD = 2 * (DIF - DEA)
        """
        self.calDIF()
        self.calDEA()
        macd = 2 * (self.indicator['dif'] - self.indicator['dea'])
        macd = DataFrame(macd)
        macd.columns = ['macd']
        self.indicator = self.indicator.join(macd)

    """
        布林线指标，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        中轨线 = N日的移动平均线
        上轨线 = 中轨线 + 两倍的标准差
        下轨线 = 中轨线 - 两倍的标准差
        策略：股价高于区间，卖出；股价低于，买入
    """
    def calSMA(self, x, n, m):
        """
            SMA(X,N,M)，求X的N日移动平均，M为权重。算法：若Y=SMA(X,N,M) 则 Y=(M*X+(N-M)*Y')/N,其中Y'表示上一周期Y值，N必须大于M。
        """
        x = list(x)
        sma = [0] * len(x)               
        for i in range(n-1, len(x)):   
            sma[i] = (x[i] * m + sma[i-1] * (n-m)) * 1.0 / n
        sma = DataFrame(sma)
        sma.columns = ['sma']
        self.indicator = self.indicator.join(sma)

    def df_round(self, data, col, decimals=2):
        """
            对 data 对几个 columns 取小数点精度
        """
        data = [0] * len(data)
        for i in data:
            data[i] = data[i].round(decimals)
        return data

    def calBOLL(self):
        """
            计算boll
        """
        self.dataset['ma_20'] = pd.rolling_mean(self.dataset['close'], 20)
        tmp = [0] * len(self.dataset)
        for i in range(19, len(self.dataset)):
            tmp[i] = self.dataset['close'][max(i-19, 0):i+1].std()
        data_std = DataFrame(tmp)
        data_std.columns = ['ma_std']
        self.indicator = self.indicator.join(data_std)
        data_boll = DataFrame(self.dataset['ma_20'])
        data_boll.columns = ['midboll']
        self.indicator = self.indicator.join(data_boll)
        data_upboll = DataFrame(self.dataset['ma_20'] + 2 * self.indicator['ma_std'])
        data_upboll.columns = ['upboll']
        self.indicator = self.indicator.join(data_upboll)
        data_lowboll = DataFrame(self.dataset['ma_20'] - 2 * self.indicator['ma_std'])
        data_lowboll.columns = ['lowboll']
        self.indicator = self.indicator.join(data_lowboll)

        # data = self.df_round(self.indicator['ma_std', 'midboll', 'upboll', 'lowboll'], 4, 2)
        return self.indicator

    
    def getIndicator(self, indicatornames):
        """
            返回指标集，方便模型操作
            默认返回所有
        """
        return self.indicator[indicatornames]
        

class ModelEngine(object):

    def __init__(self, df):
        self.dataset = df
        self.label = DataFrame([]) 
        self.discrete_attri = DataFrame([])
        self.contin_attri = DataFrame([])
        self.dataset_test = DataFrame([])
        self.dataset_train = DataFrame([])
        self.label_train = DataFrame([])
        self.label_test = DataFrame([])
        #### 以下为元组存储需要处理的DataFrame的列名 ####
        self.discrete = ("year", "month", "day", "hour", "minute") 
        self.continus = ("open", "high", "close", "low", "volume", "p_change", "ma5", "ma10", "ma20", "v_ma5", "v_ma10", "v_ma20", "turnover", "price_change")
        self.column_name = ("open", "high", "close", "low", "volume", "p_change", "ma5", "ma10", "ma20", "v_ma5", "v_ma10", "v_ma20", "turnover", "open")
        self.dataset_column = ("year", "month", "day", "hour", "minute", "open", "close", "high", "close", "low", "volume", "p_change", "ma5", "ma10", "ma20", "v_ma5", "v_ma10", "v_ma20", "turnover")
        self.delete_column = ()
        

    def dateParse(self):
        date = self.dataset['date']
        date_list = [time.strptime(x, "%Y-%m-%d %H:%M:%S") for x in date]
        Date = []
        year = [x.tm_year for x in date_list]
        month = [x.tm_mon for x in date_list]
        day = [x.tm_mday for x in date_list]
        hour = [x.tm_hour for x in date_list]
        minute = [x.tm_min for x in date_list]
        Date = [year, month, day, hour, minute]
        data = DataFrame(Date).T
        data.columns = list(self.discrete)
        data_tmp = self.dataset.drop(['date'], axis=1)
        new_data = data.join(data_tmp)
        self.dataset = new_data
        self.discrete_attri = data

    def dateParseDay(self):
        date = self.dataset['date']
        date_list = [time.strptime(x, "%Y-%m-%d") for x in date]
        Date = []
        year = [x.tm_year for x in date_list]
        month = [x.tm_mon for x in date_list]
        day = [x.tm_mday for x in date_list]
        Date = [year, month, day]
        data = DataFrame(Date).T
        data.columns = list(self.discrete)
        data_tmp = self.dataset.drop(['date'], axis=1)
        new_data = data.join(data_tmp)
        self.dataset = new_data
        self.discrete_attri = data
        
    # 离散特征值进行独热编码
    def oneHot(self):
        data_tmp = self.dataset
        data_tmp = data_tmp[list(self.discrete)]
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(data_tmp)
        data = encoder.transform(data_tmp).toarray()
        data = DataFrame(data)
        self.discrete_attri = data

    # 连续属性进行两种归一化，神经网络可用
    def scaler_Meanstd(self):
        """
            mean std
        """
        data_tmp = self.dataset
        data_tmp = data_tmp[list(self.continus)]
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_tmp)
        data = scaler.transform(data_tmp)
        data = DataFrame(data)
        data.columns = list(self.continus)
        self.contin_attri = data

    def scaler_Fixed(self):
        """
          Min Max
        """
        data_tmp = self.dataset
        data_tmp = data_tmp[list(self.continus)]
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data_tmp)
        data = scaler.transform(data_tmp)
        data = DataFrame(data)
        data.columns = list(self.continus)
        self.contin_attri = data
    
    def attriConcat(self):
        # self.oneHot()    没有必要进行独热编码
        self.scaler_Fixed()
        data = self.discrete_attri.join(self.contin_attri)
        self.dataset = data

    def set_Y(self, name):
        # price_change 作为Y，需要前移
        data_tmp = self.dataset[name]
        data_label = data_tmp[0:len(data_tmp)-1]    
        data_label_fin = data_label.reset_index(drop=True)

        # 去掉第一行
        data_tmp = self.dataset[1:]
        data_data = data_tmp.drop([name], axis=1)
        # 重新设置下标,因为join是索引连接
        data_data_fin = data_data.reset_index(drop=True)  
        self.dataset = data_data_fin.join(data_label_fin)
    
    
    def add_X(self, names, xSeries):
        """
            将指标集中的指标加入X属性中
        """
        xSeries.columns = names
        self.dataset = self.dataset.join(xSeries)

    def del_X(self):
        """
            删掉无用属性
        """
        data = self.dataset.drop(list(self.delete_column), axis=1)  # 删除这些属性
        self.dataset = data

    def split_Data(self, labelname):
        self.label = self.dataset[labelname]
        self.dataset = self.dataset[list(self.dataset_column)]
        
        # 随机划分数据集
        # data_train, data_test, label_train, label_test = train_test_split(self.dataset, self.label, test_size=0.3, random_state=0)

        # 每5个训练数据取一个测试数据
        data_train = []
        label_train = []
        data_test = []
        label_test = []
        index = 0
        self.dataset = np.array(self.dataset)
        for x in self.dataset:
            if index == 4:
                data_test.append(x)
                index = 0
            else:
                data_train.append(x)
                index = index + 1
        self.label = np.array(self.label)
        for x in self.label:
            if index == 4:
                label_test.append(x)
                index = 0
            else:
                label_train.append(x)
                index = index + 1
        data_train = DataFrame(data_train)
        data_test = DataFrame(data_test)
        label_train = DataFrame(label_train)
        label_test = DataFrame(label_test)
        # 统一操作, 适用于随机划分训练集
        # data_train = DataFrame(data_train, dtype=np.float)
        # data_test = DataFrame(data_test, dtype=np.float)
        # label_train = DataFrame(label_train, dtype=np.float)
        # label_test = DataFrame(label_test, dtype=np.float)
        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.label_test = label_test

    
    def choose_model(self, modelname):
        #### 决策树 ####
        if modelname == "DecisionTree":
            res = tree.DecisionTreeRegressor()
        #### 线性回归 ####
        elif modelname == "Linear":
            res = linear_model.LinearRegression()
        #### 支持向量机 ####
        elif modelname == "SVM":
            res = svm.SVR()
        #### KNN ####
        elif modelname == "KNN":
            res = neighbors.KNeighborsRegressor()
        #### RF ####
        elif modelname == "RF":
            res = ensemble.RandomForestRegressor(n_estimators=10)  # 参数待定
        #### Adaboost ####
        elif modelname == "Adaboost":
            res = ensemble.AdaBoostRegressor(n_estimators=50)  # 参数待定
        #### 梯度提升决策树　####
        elif modelname == "GBDT":
            res = ensemble.GradientBoostingRegressor()
        #### 袋装 ####
        elif modelname == "Bagging":
            res = BaggingRegressor()
        #### 不太清楚 ####
        elif modelname == "ExtraTree":
            res = ExtraTreeRegressor()
        #### 神经网络 ####
        elif modelname == "NN":
            res = MLPRegressor()
        return res

    # 通用模型处理步骤
    def model_Processing(self, modelname):
        model = self.choose_model(modelname)
        model.fit(self.data_train, self.label_train)
        score = model.score(self.data_test, self.label_test)
        print(score)
        self.model = model
        result = model.predict(self.data_test)
        plt.figure()
        plt.plot(self.label_test)
        plt.plot(result)
        # 计算两者之间相关系数
        plt.legend()
        plt.show()
    
    def model_application(self, dataset):
        model = self.model
        res = model.predict(dataset)
        return(res)


if __name__ == "__main__":

    """
        预处理，转换为我们内部一个更快格式的数据并输出
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the directory of the .csv file")
    parser.add_argument("code", type=str, help="the stock code you want to predict")
    parser.add_argument("-m", "--method", type=str,
                        default='NeuralNetwork', help="the method used to do this task")
    parser.add_argument("-r", "--ratio", type=float,
                        default=0.2, help="the validate data set ratio")
    parser.add_argument("-s", "--train_set", type=float, default=1.0, help="the minimun correlation of stocks we use in train set, 1 means the train set doesn't includes other stocks")
    parser.add_argument("-p", "--preprocess", type=int, default=1,  help="preprocessing or not")
    args = parser.parse_args()

    
    filename = "5min/000001.csv"
    Y = "price_change"
    frData = FastResearchData()
    frData.loadFromCSV_Pandas(filename)
    dataSet = frData.getDataFrame()

    """
        这里存放了指标计算方法
        以将各种指标分成不同的类，以此来管理各种各样的不同分类指标
    """
    indicatorname = ['log_price', 'ema_12', 'ema_26', 'dif', 'dea', 'macd', 'upboll', 'lowboll']
    xmIG = IndicatorGallexy(dataSet)
    xmIG.calLogPrice()
    xmIG.calMACD()
    xmIG.calBOLL()
    indicators = xmIG.getIndicator(indicatorname)

    """
        ModelEngine是一个管理训练和评估过程的类
        可以在ModelEngine中选择不同的模型，以及不同的训练方法，以及不同的变量
    """
    modelname = "KNN"
    model = ModelEngine(dataSet)
    model.dateParse()
    model.add_X(indicatorname, indicators)
    model.del_X()
    model.attriConcat()
    model.set_Y(Y)
    # 划分数据集为训练集和验证集，方便测试
    model.split_Data(Y)
    # 选择模型并处理
    model.model_Processing(modelname)
    # 模型运用
    # dataset = ...
    # Out = model.model_application(dataset)
