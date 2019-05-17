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
import  pandas_profiling 
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
1. 取前 N日的最高价序列与最低价序列。
2. 将两列数据按式（1）的模型进行 OLS 线性回归。
3. 将拟合后的 beta 值作为当日 RSRS斜率指标值。
另一种则为将斜率标准化，取其标准分作为指标值。 当日标准分指标的计算方式:
1. 取前 M日的斜率时间序列。
2. 以此样本计算当日斜率的标准分。 
3. 将计算得到的标准分 z-score作为当日 RSRS标准分指标值。 
'''
#方法一,当日斜率指标的计算方式
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
    return df1.reset_index()
#方法二：
'''
）总体数据的均值（μ）

     在上面的例子中，总体可以是整个班级的平均分，也可以是全市、全国的平均分。

2）总体数据的标准差（σ）

     这个总体要与1）中的总体在同一个量级。

3）个体的观测值（x）
'''
#方法二,取其标准分作为指标值。 当日标准分指标的计算方式
def slop_method2(df,n,m):#N日的最高价序列与最低价序列,取前 M日的斜率时间序列。
    df1=slope_method1(df,n)#求出斜率
    length=df1.shape[0]
    z_scores=[]
    df1=df1.reset_index()#重建索引
    df2=df1[m+1:]
    for j in range(0,length-m-1):
        m_day_df=df1.iloc[j:j+m]
        print()
        slope=m_day_df['day_slope'].values
        avg=np.mean(slope)
        std=np.std(slope)
        print(avg,std)
        z_score=(df2['day_slope'].values[j]-avg)/std
        print(z_score)
        z_scores.append(z_score)
    df2['z_score']=z_scores
    return df2
    
def ols_regression(x,y,eps):#ols回归
    X=sm.add_constant(x) 
    model = sm.OLS(y-eps,X)
    results = model.fit() 
    return(results.params) 
def apply_strategic1(n_df,slope1,slope2):#回测1
    buy_sell=[]#是否卖出或者买入或者其他状态
#     origin_price=n_df.iloc[0].closed_price_today
    day_slope=n_df['day_slope'].values#斜率
    state='空仓'
    for i in day_slope:
        if i>slope1 and state=='观望':
            buy_sell.append('买入')
            state='持仓'
        elif i<slope2 and state=='持仓':
            buy_sell.append('卖出')
            state='空仓'
        else :
            buy_sell.append(state)
    n_df['buy_sell']=buy_sell
    return n_df
#策略2
def apply_strategic2(mn_df,z_score1,z_score2):#回测
    buy_sell=[]#是否卖出或者买入或者其他状态
#     origin_price=n_df.iloc[0].closed_price_today
    day_slope=mn_df['z_score'].values#斜率
    state='空仓'
    for i in day_slope:
        if i>z_score1 and state=='观望':
            buy_sell.append('买入')
            state='持仓'
        elif i<z_score2 and state=='持仓':
            buy_sell.append('卖出')
            state='空仓'
        else :
            buy_sell.append(state)
    mn_df['buy_sell']=buy_sell
    return mn_df
  
def calcu_net_value(transaction_df,method='slope'):#计算净值
    strategic='slope'
    if strategic==method:
        tr_df=apply_strategic1(transaction_df,1,0.75)
    else:
        tr_df=apply_strategic2(transaction_df,0.75,-0.75)
    Net_value=[]
    price_buy_sell=tr_df[['closed_price_today','buy_sell']]
    origin_price=price_buy_sell.closed_price_today.iloc[0]
    net_value=1
    for index,j in price_buy_sell.iterrows():
        if(j['buy_sell']!='空仓'):
            net_value*=j['closed_price_today']/origin_price
            Net_value.append(net_value)
        else:
            net_value1=net_value*j['closed_price_today']/origin_price 
            Net_value.append(net_value1)       
    print(Net_value)                     
    tr_df[method+'_net_value']=Net_value
    return tr_df
    tr_df=apply_strategic2(transaction_df,1,-1)
def base_net_value(df1):
    #     以第一交易日2009年1月5日收盘价为基点，计算净值
    df_new=df1.closed_price_today/df.closed_price_today.iloc[0]
    print(df_new)
    #将上述股票在回测期间内的净值可视化
    df_new.plot(figsize=(16,10))
    #图标题
    plt.title(u'stock change',fontsize=12)
    #设置x轴坐标
    my_ticks = df1.date
    plt.xticks(np.arange(len(my_ticks)),my_ticks,fontsize=1)
    #去掉上、右图的线
    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()   
       
if __name__=='__main__':
    file_path='./data/000905.SH.mat'
    df=mat_to_df(file_path)
    df.date=df.date.apply(lambda i : str(int(i)))
#     df1=slope_method1(df,18)
#     plt.plot(df1['lowest_price_today'].values,df1['highest_price_today'])
#     plt.show()
#     df1.date=df1.date.apply(lambda i : str(int(i)))
#     print(df1.date)
#     df1.reset_index().to_csv('./data/000905.SH_1.csv',)
    df1=slope_method1(df,18)
#     df1.day_slope=df1.day_slope.apply(lambda i:round(i,2))
#     pfr=pandas_profiling.ProfileReport(pd.DataFrame(df1.day_slope).reset_index())
#     pfr.to_file('report.html')#生成斜率报告
    '''
     RSRS 斜率指标交易策略为： 1. 计算 RSRS 斜率。 2. 如果斜率大于 1，则买入持有。 3. 如果斜率小于 0.75，则卖出手中持股平仓。 
     
    '''
    df2=slop_method2(df,18,600)
#     df2.z_score=df2.z_score.apply(lambda i:round(i,2))
#     pfr=pandas_profiling.ProfileReport(pd.DataFrame(df2.z_score).reset_index())
#     pfr.to_file('z_score_report.html')#生成标准分报告
    '''
            则 RSRS 标准分交易策略为： 1. 根据斜率计算标准分（参数 N=18,M=600）。 2. 如果标准分大于 S（参数 S=1），则买入持有。 3. 如果标准分小于-S，则卖出平仓。
    '''
#     exchange_detail1=apply_strategic1(df1,1,0.75)
    
#     base_net_value(df1)
    calcu_net_value(df1[600+1:],method='slope')
    calcu_net_value(df2,method='slope')
