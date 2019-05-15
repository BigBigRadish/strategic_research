# -*- coding: utf-8 -*-
'''
Created on 2019年5月15日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
#工具模块
import pandas as  pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.io import loadmat#用于加载mat文件
def mat_to_df(file_path):
    data_m= loadmat(file_path)#mat-->dict字典。
    # print(data_m.keys())#其中有很多key，注意找自己数据所在的那个 
    data_load = data_m['prc__']#我需要的是这个，这是一个2998*10的矩阵 
    print(data_m['__header__'])
    data_load = pd.DataFrame(data_load,columns=("date","closed_price_yesterday","open_price_today","highest_price_today","lowest_price_today","closed_price_today","trade_volume","trade_amount"))#给每列加上一个名字或者叫特征名，此时data_load就是一个dataframe类型了。 
    # print(data_load.head())#打印该df的前五行 
    # print(data_load.shape)
    return data_load
#
'''
high= alpha + beta*low + epsilon， epsilon ~ N(0,sigma) （1)
一种即是直接利用斜率的本身作为指标值。 
当日斜率指标的计算方式：
1. 取前 N 日的最高价序列与最低价序列。
2. 将两列数据按式（1）的模型进行 OLS 线性回归。
3. 将拟合后的 beta 值作为当日 RSRS斜率指标值。
另一种则为将斜率标准化，取其标准分作为指标值。 当日标准分指标的计算方式:
1. 取前 M 日的斜率时间序列。
2. 以此样本计算当日斜率的标准分。 
3. 将计算得到的标准分 z 作为当日 RSRS标准分指标值。 
'''
def slope_method1(df,n):
    length=df.shape[0]
    slope_param=[]
    df1=df.iloc[n:]
    for i in range(0,length-n):
        n_day_df=df.iloc[i:i+n]
        x=n_day_df['lowest_price_today'].values
        y=n_day_df['highest_price_today'].values
        eps= np.random.normal(size=n)
        param=ols_regression(x,y,eps)
        slope_param.append(param[1])
    df1['day_slope']=slope_param
    return df1
def ols_regression(x,y,eps):
    X=sm.add_constant(x) 
    model = sm.OLS(y-eps,X)
    results = model.fit() 
    return(results.params) 
if __name__=='__main__':
    file_path='./data/000016.SH.mat'
    df=mat_to_df(file_path)
    df1=slope_method1(df,18)
#     plt.plot(df1['lowest_price_today'].values,df1['highest_price_today'])
#     plt.show()
    df1.date=df1.date.apply(lambda i : str(int(i)))
    print(df1.date)
    df1.to_csv('./data/000016.SH_1.csv')
    
    
    
