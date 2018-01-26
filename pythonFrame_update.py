import argparse
import csv
import numpy as np
import argparse
import time
import math
import pickle
import pymysql
import sqlalchemy
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
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
        打开多个文件，文件名以list传入，将多个文件中的数据整合为一个
    """
    def __init__(self):
        ''' stock_dict:股票字典'''
        self.stockdict = {}

    def loadFromDataFrame(self, stockcode, df):
        '''
            直接读取df
        '''
        if stockcode not in self.stockdict.keys():
            self.stockdict[stockcode] = df
        else:
            pd.concat([self.stockdict[stockcode], df])  

    def loadFromCSV(self, stockcodeslist, filenameslist):
        """
            加载csv:使用pandas
        """
        for (stockcode, filename) in zip(stockcodeslist, filenameslist):
            data = pd.read_csv(filename)  
            data = DataFrame(data)
            self.loadFromDataFrame(stockcode, data)
    
    def loadFromPickle(self, stockcodeslist, filenameslist):
        '''
            加载pickle
            python的pickle模块实现了基本的数据序列和反序列化。序列化通过dump()方法实现，反序列化通过load()方法实现
            通过pickle的序列化操作能够将程序中运行的对象保存到文件中去，永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象
        '''
        for (stockcode, filename) in zip(stockcodeslist, filenameslist):
            pickle_file = open(filename, 'rb')
            data = pickle.load(pickle_file)
            data = DataFrame(data)
            self.loadFromDataFrame(stockcode, data)

    def loadFromDB(self, hostname, username, password, dbname, tablename, stockcode):
        '''
            连接数据库,需要提供主机名 用户名 密码 数据库名 
        '''
        con = pymysql.connect(host=hostname, user=username, password=password, database=dbname, user_unicode=True)
        sql_cmd = "SELECT * FROM" + str(tablename)
        data = pd.read_sql(sql_cmd, con)   
        data = DataFrame(data)
        self.loadFromDataFrame(stockcode, data)

    def getDataFrame(self, stockcode):
        """
            返回某只股票
        """
        return self.stockdict[stockcode]

    def getStockDictionary(self):
        """
            返回整个股票字典
        """
        return self.stockdict


class IndicatorGallexy(object):
    def __init__(self, df):
        ''' indicatordict:指标字典 '''
        self.dataset = df
        self.indicatordict = {}

    def loadIndicator(self, indicatorname, df):
        if indicatorname not in self.indicatordict.keys():
            self.indicatordict[indicatorname] = df
        else:
            pd.concat([self.indicatordict[indicatorname], df])

    def calLog(self, minuend, meiosis):
        """
            log(minuend) - log(meiosis)
        """
        minuend_price = self.dataset[minuend]
        meiosis_price = self.dataset[meiosis]
        res = []
        for (x, y) in zip(minuend_price, meiosis_price):
            tmp = math.log(y) - math.log(x)
            res.append(tmp)
        indicator = DataFrame(res)
        indicatorname = "log_" + meiosis + "_" + minuend
        indicator.columns = [indicatorname]
        self.loadIndicator(indicatorname, indicator)

    """
        MACD:指数平滑移动平均线，是从双指数移动平均线发展而来的，由快的指数平均线（EMA12）减去慢的指数移动平均线（EMA26）
        得到快线DIF，再用2*（快线DIF-DIF的九日加权移动均线DEA）得到MACD柱
    """

    def calEMA(self, shortNumber, longNumber, attriname):
        """
            计算移动平均值，快速移动平均线为12日，慢速移动平均线为26日
            快速：EMA[i] = EMA[i-1] * (short - 1)/(short + 1) + close * 2 / (short + 1) 
            慢速：EMA[i] = EMA[i-1] * (long - 1)/(long + 1) + close * 2 / (long + 1) 
        """
        ema_short = [self.dataset[attriname][0]] * len(self.dataset)
        ema_long = [self.dataset[attriname][0]] * len(self.dataset)
        for i in range(1, len(self.dataset)):
            ema_short[i] = ema_short[i-1] * (shortNumber-1)/(shortNumber+1) + self.dataset[attriname][i] * 2/(shortNumber+1)
            ema_long[i] = ema_long[i-1] * (longNumber-1)/(longNumber+1) + self.dataset[attriname][i] * 2/(longNumber+1)
        ema_short = DataFrame(ema_short)
        ema_shortname = "ema" + str(shortNumber)
        ema_short.columns = [ema_shortname]  
        ema_long = DataFrame(ema_long)
        ema_longname = "ema" + str(longNumber)
        ema_long.columns = [ema_longname]
        self.loadIndicator(ema_shortname, ema_short)
        self.loadIndicator(ema_longname, ema_long)

    def calDIF(self, emashortname, emalongname):
        """
            DIF为离差值，涨势中，离差值会变得越来越大，跌势中，离差值会变得越来越小
            DIF = EMA(short) - EMA(long)
        """
        dif = self.indicator[emashortname] - self.indicator[emalongname]
        dif = DataFrame(dif)
        difname = "dif"
        dif.columns = [difname]
        self.loadIndicator(difname, dif)

    def calDEA(self, difname, n):
        """
            计算DEA差离平均值
            DEA[i] = DEA[i-1] * (n-1) / (n+1) + DIF[i] * 2 / (n+1)
            其中n为多少日
        """
        dea = [self.indicator[difname][0]] * len(self.dataset)
        for i in range(1, len(self.dataset)):
            dea[i] = dea[i - 1] * (n - 1) / (n + 1) + self.indicator[difname][i] * 2 / (n + 1)
        dea = DataFrame(dea)
        deaname = "dea"
        dea.columns = deaname
        self.loadIndicator(deaname, dea)

    def calMACD(self, difname, deaname):
        """
            计算MACD指数平滑移动平均线
            MACD = 2 * (DIF - DEA)
        """
        macd = 2 * (self.indicator[difname] - self.indicator[deaname])
        macd = DataFrame(macd)
        macdname = "macd"
        macd.columns = [macdname]
        self.loadIndicator(macdname, macd)

    def calBOLL(self, indicatorname, numberofdays):
        """
            布林线指标，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
            中轨线 = N日的移动平均线
            上轨线 = 中轨线 + 两倍的标准差
            下轨线 = 中轨线 - 两倍的标准差
            策略：股价高于区间，卖出；股价低于，买入
        """
        ma_days = DataFrame([])
        ma_days = pd.rolling_mean(self.dataset[indicatorname], numberofdays)
        tmp = [0] * len(self.dataset)
        for i in range(19, len(self.dataset)):
            tmp[i] = self.dataset[indicatorname][max(i-(numberofdays-1), 0):i+1].std()
        data_std = DataFrame(tmp)
        data_stdname = 'ma_std'
        data_std.columns = [data_stdname]
        self.loadIndicator(data_stdname, data_std)

        data_boll = DataFrame(ma_days)
        data_bollname = 'midboll'
        data_boll.columns = [data_bollname]
        self.loadIndicator(data_bollname, data_boll)

        data_upboll = DataFrame(data_boll + 2 * data_std)
        data_upbollname = 'upboll'
        data_upboll.columns = [data_upbollname]
        self.loadIndicator(data_upbollname, data_upboll)

        data_lowboll = DataFrame(data_boll + 2 * data_std)
        data_lowbollname = 'lowboll'
        data_lowboll.columns = [data_lowbollname]
        self.loadIndicator(data_lowbollname, data_lowboll)

    
    def getIndicator(self, indicatorname):
        """
            返回指
        """
        return self.indicatordict[indicatorname]


class DatasetPreProcessing(object):
    """
        针对给定的五分钟数据集进行预处理
    """
    def __init__(self, df):
        self.dataset = df
        self.attributedict = {}
    
    def loadAttribute(self, attriname, df):
        if attriname not in self.attributedict.keys():
            self.attributedict[attriname] = df
        else:
            pd.concat([self.attributedict[attriname], df])

    def dateParseMinute(self, attriname):
        """
            划分时间特征：处理五分钟数据
        """
        date = self.dataset[attriname]
        date_list = [time.strptime(x, "%Y-%m-%d %H:%M:%S") for x in date]
        year = [x.tm_year for x in date_list]
        month = [x.tm_mon for x in date_list]
        day = [x.tm_mday for x in date_list]
        hour = [x.tm_hour for x in date_list]
        minute = [x.tm_min for x in date_list]
        data = DataFrame([year, month, day, hour, minute]).T
        attributenames = ['year', 'month', 'day', 'hour', 'minute']
        data.columns = attributenames
        for (attributename, i) in zip(attributenames, data):
            self.loadAttribute(attributename, data[attributename])

    def dateParseDay(self, attriname):
        """
            划分时间特征：处理一天数据
        """
        date = self.dataset[attriname]
        date_list = [time.strptime(x, "%Y-%m-%d") for x in date]
        year = [x.tm_year for x in date_list]
        month = [x.tm_mon for x in date_list]
        day = [x.tm_mday for x in date_list]
        data = DataFrame([year, month, day]).T
        attributesname = ['year', 'month', 'day']
        data.columns = attributesname
        for (attributename, i) in zip(attributenames, data):
            self.loadAttribute(attributename, data[attributename])
    
    def oneHot(self, attrinameslist):
        """
            离散属性进行独热编码
        """
        data_tmp = self.dataset[attrinameslist]
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(data_tmp)
        data = encoder.transform(data_tmp).toarray()
        data = DataFrame(data)
        attributesnames = []
        for x in range(len(data.columns)):
            attributesnames.append("one_hot"+str(x))
        data.columns = attributesnames
        for (attributename, i) in zip(attributenames, data):
            self.loadAttribute(attributename, data[attributename])

    def scalerFixed(self, attrinameslist):
        """ 
            连续属性进行归一化： 均值方差归一化
        """
        data_tmp = self.dataset[attrinameslist]
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_tmp)
        data = scaler.transform(data_tmp)
        data = DataFrame(data)
        attributesnames = attrinameslist
        data.columns = attributesnames
        for (attributename, i) in zip(attributenames, data):
            self.loadAttribute(attributename, data[attributename])

    def scalerMeanstd(self, attrinameslist, upbound, downbond):
        """
            连续属性进行归一化： 最大最小归一化
        """
        data_tmp = self.dataset
        scaler = preprocessing.MinMaxScaler(feature_range=(upbound, downbond))
        scaler.fit(data_tmp)
        data = scaler.transform(data_tmp)
        data = DataFrame(data)
        attributesnames = attrinameslist
        data.columns = attributesnames
        for (attributename, i) in zip(attributenames, data):
            self.loadAttribute(attributename, data[attributename])

    def correctDataset(self, attriname):
        """
            处理数据集,price_change前移
        """
        ''' price_change 作为Y，需要前移 '''
        data_tmp = self.dataset[attriname]
        data_label = data_tmp[0:len(data_tmp)-1]    
        data_label_fin = data_label.reset_index(drop=True)

        ''' 去掉第一行 '''
        data_tmp = self.dataset[1:]
        data_data = data_tmp.drop([attriname], axis=1)
        data_data_fin = data_data.reset_index(drop=True)  
        self.dataset = data_data_fin.join(data_label_fin)
        self.loadAttribute(attriname, data_data_fin)

    def getAttribute(self, attributename):
        """"
            取出单列特征
        """
        if attribute in self.attributedict.keys():
            return self.attributedict[attributename]
        else:
            print('Not Exists!')
    
    def getDataset(self):
        """
            返回整个数据集
        """
        return DataFrame(self.dataset)


class ModelEngine(object):

    def __init__(self, df):
        self.dataset = df
        self.label = DataFrame([]) 
        self.dataset_test = DataFrame([])
        self.dataset_train = DataFrame([])
        self.label_train = DataFrame([])
        self.label_test = DataFrame([])
    
    def set_Y(self, name):
        """
            选择决定属性Y， 将其他归为dataset
        """
        self.label = self.dataset[name]
        dataset_column = []
        for x in self.dataset.columns:
            if x != name:
                dataset_column.append(x)
        self.dataset = self.dataset[dataset_column]
    
    def add_X(self, xSeries):
        """
            将指标集中的指标加入X属性中
        """
        self.dataset = self.dataset.join(xSeries)

    def del_X(self, attributenameslist):
        """
            删掉多个无用属性
        """
        data = self.dataset.drop(attributenameslist, axis=1)  # 删除这些属性
        self.dataset = data

    def split_Data(self):
        """
            划分数据集
        """
        data_train, data_test, label_train, label_test = train_test_split(self.dataset, self.label, test_size=0.3, random_state=0)
        # data_train.reset_index(drop=True)
        # data_test.reset_index(drop=True)
        # label_train.reset_index(drop=True)
        # label_test.reset_index(drop=True)
        # data_train = DataFrame(data_train, dtype=np.float)
        # data_test = DataFrame(data_test, dtype=np.float)
        # label_train = DataFrame(label_train, dtype=np.float)
        # label_test = DataFrame(label_test, dtype=np.float)
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
        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.label_test = label_test

    
    def choose_model(self, modelname):
        ### 决策树 ###
        if modelname == "DecisionTree":
            res = tree.DecisionTreeRegressor()
        ### 线性回归 ###
        elif modelname == "Linear":
            res = linear_model.LinearRegression()
        ### 支持向量机 ###
        elif modelname == "SVM":
            res = svm.SVR()
        ### KNN ###
        elif modelname == "KNN":
            res = neighbors.KNeighborsRegressor()
        ### RF ###
        elif modelname == "RF":
            res = ensemble.RandomForestRegressor(n_estimators=10)  # 参数待定
        ### Adaboost ###
        elif modelname == "Adaboost":
            res = ensemble.AdaBoostRegressor(n_estimators=50)  # 参数待定
        ### 梯度提升决策树　###
        elif modelname == "GBDT":
            res = ensemble.GradientBoostingRegressor()
        ### 袋装 ###
        elif modelname == "Bagging":
            res = BaggingRegressor()
        ### 不太清楚 ###
        elif modelname == "ExtraTree":
            res = ExtraTreeRegressor()
        ### 神经网络 ###
        elif modelname == "NN":
            res = MLPRegressor()
        return res

    def model_Processing(self, modelname):
        """
            模型处理通用步骤
        """
        model = self.choose_model(modelname)
        model.fit(self.data_train, self.label_train)
        score = model.score(self.data_test, self.label_test)
        self.model = model
        result = model.predict(self.data_test)
        print(score)
        plt.figure()
        plt.plot(self.label_test, color="red")
        plt.plot(result, color="black")
        plt.legend()
        plt.show()
    
    def model_application(self, dataset):
        model = self.model
        res = model.predict(dataset)
        return(res)

if __name__ == "__main__":

    """
        命令行解析
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument("filename", type=str, help="the directory of all the .csv file")
    # parser.add_argument("decision_attribute", type=str, help="the attribute you want to predict")
    # parser.add_argument("-m", "--model", type=str, default='KNN', help="the mmodel used to do this task")
    # parser.add_argument("-p", "--preprocess", type=int, default=1,  help=" preprocessing or not")
    # parser.add_argument("-i", "--indicatorcalculate", type=int, default=1,  help=" calculate indicator")
    # args = parser.parse_args()

    """
        加载数据
    """
    stockcodeslist = ['000001']
    filenameslist = ["5min/000001.csv"]
    decision_attribute = "price_change"
    frData = FastResearchData()
    frData.loadFromCSV(stockcodeslist, filenameslist)
    stock = frData.getDataFrame(stockcodeslist[0])

    """
        这里存放了指标计算方法
        以将各种指标分成不同的类，以此来管理各种各样的不同分类指标
    """

    dataset = DatasetPreProcessing(stock)
    dataset.correctDataset(decision_attribute)

    newDataset = dataset.getDataset()

    indicatorname = "log_close_open"
    xmIG = IndicatorGallexy(newDataset)
    xmIG.calLog('open', 'close')
    indicators = xmIG.getIndicator(indicatorname)

    """
        ModelEngine是一个管理训练和评估过程的类
        可以在ModelEngine中选择不同的模型，以及不同的训练方法，以及不同的变量
    """
    modelname = "KNN"
    model = ModelEngine(newDataset)
    model.set_Y(decision_attribute)
    model.add_X(indicators)
    model.del_X('date')
    model.split_Data()
    model.model_Processing(modelname)
    # 模型运用
    # dataset = ...
    # Out = model.model_application(dataset)
