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
from decimal import *
import threading
import functools
import time

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
    df1_=df.iloc[n:]
    for i in range(0,length-n):
        n_day_df=df.iloc[i:i+n]
        x=n_day_df['lowest_price_today'].values
        y=n_day_df['highest_price_today'].values
        eps= np.random.normal(size=n)
        param=ols_regression(x,y,eps)
        slope_param.append(param[1])
    df1_['day_slope']=slope_param
    df1_['day_slope']=df1_['day_slope'].apply(lambda i:round(i,2))#注意精度
    df1_=df1_.reset_index()
    return df1_.copy()
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
    df1_=slope_method1(df,n)#求出斜率
    length=df1_.shape[0]
    z_scores=[]
    df2=df1_[m:].reset_index()
    for j in range(0,length-m):
        m_day_df=df1_.iloc[j:j+m]
        slope=m_day_df['day_slope'].values
        avg=round(np.mean(slope),2)
        std=round(np.std(slope),2)
#         print(avg,std)
#         print(df2['day_slope'].values)
#         print(df2['day_slope'].values[j])
        z_score=round((df2['day_slope'].values[j]-avg)/std,2)
#         print(z_score)
        z_scores.append(z_score)
    df2['z_score']=z_scores
    df2['z_score']=df2['z_score'].apply(lambda i:round(i,2))#多核加速
#     df2=df2
    return df2.copy()
    
def ols_regression(x,y,eps):#ols回归
    X=sm.add_constant(x) 
    model = sm.OLS(y-eps,X)
    results = model.fit() 
    return(results.params) 
def apply_strategic1(n_df,slope1,slope2):#回测1
    buy_sell=[]#是否卖出或者买入或者其他状态
#     origin_price=n_df.iloc[0].closed_price_today
    day_slope=n_df['day_slope'].values#斜率
    state=u'空仓'
    for i in day_slope:
        if i>slope1 and state==u'空仓':
            buy_sell.append(u'买入')
            state=u'持仓'
        elif i<slope2 and state==u'持仓':
            buy_sell.append(u'卖出')
            state=u'空仓'
        else :
            buy_sell.append(state)
    n_df['buy_sell']=buy_sell
    return n_df.copy()
#策略2
def apply_strategic2(mn_df,z_score1,z_score2):#回测2
    buy_sell=[]#是否卖出或者买入或者其他状态
#     origin_price=n_df.iloc[0].closed_price_today
    z_score=mn_df['z_score'].values#斜率
    state=u'空仓'
    for i in z_score:
        if i>z_score1 and state==u'空仓':
            buy_sell.append(u'买入')
            state=u'持仓'
        elif i<z_score2 and state==u'持仓':
            buy_sell.append(u'卖出')
            state=u'空仓'
        else :
            buy_sell.append(state)
#     print(buy_sell)
    mn_df['buy_sell']=buy_sell
    return mn_df.copy()

def calcu_net_value(transaction_df,method):#计算净值
    strategic='slope'
    if strategic==method:
        tr_df=apply_strategic1(transaction_df,0.75,0.6)
    else:
        tr_df=apply_strategic2(transaction_df,1,-1)
#     print(tr_df)
    Net_value=[]
    price_buy_sell=tr_df[['closed_price_today','buy_sell']].copy()
#     print(price_buy_sell)
    origin_price=price_buy_sell.closed_price_today.iloc[0]
#     print(origin_price)
#     print(price_buy_sell.closed_price_today)
#     print(origin_price)
    net_value=1
    for index,j in price_buy_sell.iterrows():
#         print(index,j)
#         print(j['buy_sell'])
        if(j['buy_sell'] not in [u'空仓']):#不是空仓正常算净值
#             print(origin_price)
            net_value=(j['closed_price_today']/origin_price)
#             print(net_value)
            Net_value.append(net_value)
        else:
#             print(net_value)
            net_value1=net_value*(j['closed_price_today']/origin_price)
#             print(net_value1)
            Net_value.append(net_value1)
        if (j['buy_sell']==u'买入'):
            origin_price=j['closed_price_today']
#             print(origin_price)
        else:
            pass     
#     print(Net_value)                     
    tr_df['net_value']=Net_value
#     print(tr_df[method+'_net_value'])
#     print(Net_value)
#     tr_df.to_csv('000905.SH_z_score_net.csv')
    return tr_df
#画净值图
def plot_net_value(base_net,slope_net,z_score_net,date):
    z_net=list(z_score_net)
    print(len(slope_net))
    x=np.arange(len(z_net))
    base_net=list(base_net)
    slope_net=list(slope_net)
    fig, ax = plt.subplots(1, 1)
    #将上述股票在回测期间内的净值可视化
    plt.plot(date,slope_net.net_value,color='yellow',label='slope_net',)
    plt.plot(date,base_net,color='blue',label='base_net')
    plt.plot(date,z_net.net_value,color='red',label='z_score_net')
    
    #图标题
    plt.title('net_value',fontsize=10)
    #设置x轴坐标
#     myticks=z_score_net.date
#     plt.xticks(np.arange(len(z_score_net)),z_score_net,fontsize=1,color='blue')
    #去掉上、右图的线
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::200]:
        label.set_visible(True)

    plt.legend()
    plt.ylabel('net_value')
    plt.xlabel('date')
    plt.show() 
#统计指数
def statics_index(s_df):  
    '''
年化收益，夏普比率，最大回撤，持仓总天数，交易次数，平均持仓数，获利天数，亏损天数，胜率，平均盈利率，平均亏损率，平均盈亏比，
盈利次数，亏损次数，单次最大盈利，单次最大亏损，胜率（按次），平均盈利率，平均亏损率，平均盈亏比(按次）
年收益率=[（投资内收益 / 本金）/ 投资天数] * 365 ×100%
夏普比率=*代表投资人每多承担一分风险，可以拿到几分报酬；
---单位风险所获得的超额回报率
*该比率越高，策略承担单位风险得到的超额回报率越高。
所以说夏普比率是越高越好滴..
Sharpe_ratio= R_p-R_f/sigma_p
其中，R_p为年化收益率， R_f 是无风险收益率，sigma_p为年化波动率
    '''
    ### 区间累计收益率(绝对收益率)
    total_ret=s_df['net_value']-1
    TR=pd.DataFrame(total_ret.values,columns=['累计收益率'],index=total_ret.index)
#     print(TR)
    ###年化收益率,假设一年以250交易日计算
    annual_ret=pow(1+total_ret,250/len(s_df))-1
#     print(annual_ret)
    AR=pd.DataFrame(annual_ret.values,columns=['年化收益率'],index=annual_ret.index)
    print('年化：',AR.values[-1])
    #定义成函数，减少重复工作
    def max_drawdown(df):
        md=((df.cummax()-df)/df.cummax()).max()
        return round(md,4)
    md=max_drawdown(s_df['net_value'])#最大回撤
    #最大回撤率结果：
    print('最大回撤：',md)
    #计算每日收益率
    #收盘价缺失值（停牌），使用前值代替
    net_value_1=s_df['net_value'].values
    rets=[0]
    for i in range(0,len(net_value_1)-1):
        ret=(net_value_1[i+1]-net_value_1[i])/net_value_1[i]
        rets.append(ret)
    s_df['rets']=rets
    #假设无风险收益率为年化3%
    exReturn=s_df.rets-0.03/250
    #计算夏普比率
    sharperatio=np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
    print('夏普率：',sharperatio)
    #夏普比率的输出结果
#     SHR=pd.DataFrame(sharperatio,columns=['夏普比率'])
#     print(SHR)
    
    '''
    按天
    '''
    hold_positions=s_df[s_df['buy_sell']!=u'空仓']#持仓
    #持仓总天数
    hold_positions_day=len(s_df[s_df['buy_sell']!=u'空仓'])
    print('持仓天数：',hold_positions_day)
    #交易次数
    transaction_times=len(s_df[s_df['buy_sell']==u'买入'])+len(s_df[s_df['buy_sell']==u'卖出'])
    print('交易次数：',transaction_times)
    avg_hold_positions=round(float(hold_positions_day)/transaction_times,2)#平均持仓天数
    print('平均持仓天数：',avg_hold_positions)
    reap_profit_days=1#获利天数
    loss_days=0#亏损天数
    profit_rate_day=[]#平均盈利率列表
    loss_profit_day=[]#
    for i in range(0,hold_positions_day-1):
        if (hold_positions.net_value.values[i+1]>hold_positions.net_value.values[i]):
            reap_profit_days+=1
            profit_rate_day.append(round(hold_positions.net_value.values[i+1]/hold_positions.net_value.values[i],5)-1)
        else:
            loss_days+=1
            loss_profit_day.append(1-round(hold_positions.net_value.values[i+1]/hold_positions.net_value.values[i],5))
    print('获利天数：',reap_profit_days,'亏损天数：',loss_days)
    win_rate_day=round(float(reap_profit_days)/hold_positions_day,4)#胜率
    avg_profit_rate=round(np.mean(profit_rate_day),4)#平均盈利率
    avg_loss_rate=round(np.mean(loss_profit_day),4)#平均亏损率
    avg_pl_rate=round(avg_profit_rate/avg_loss_rate,2)#平均盈亏比
    print('胜率（按天）：',win_rate_day,'平均盈利率（按天）：',avg_profit_rate,'平均亏损率（按天）：',avg_loss_rate,'平均盈亏比（按天）：',avg_pl_rate)
    '''
    按次
    '''
#     buy_net_value=hold_positions[hold_positions['buy_sell']==u'买入'].slope_net_value.values#买入净值
#     sell_net_value=hold_positions[hold_positions['buy_sell']==u'卖出'].slope_net_value.values#卖出净值
#     if(len(buy_net_value)>len(sell_net_value)):
#         buy_net_value=buy_net_value[0:-1]
#     elif (len(buy_net_value)<len(sell_net_value)):
#         sell_net_value=sell_net_value[0:-1]
#     else:
#         pass
#     
    single_rate=[]#单次盈利
    transaction_detail=hold_positions[hold_positions[hold_positions['buy_sell']!=u'空仓']['buy_sell']!='持仓'].net_value.values
    single_rate.append(transaction_detail[0]-1)
    for i in range(0,len(transaction_detail)-1):
        single_rate.append(round(transaction_detail[i+1]/transaction_detail[i]-1,4))
#     print(single_rate)
    profit_rate_times=sum(np.array(single_rate)>0)#盈利次数
    loss_rate_times=sum(np.array(single_rate)<=0)#损失次数
    print('盈利次数：',profit_rate_times,'损失次数：',loss_rate_times)
    single_max_profit,single_max_loss=np.max(single_rate),np.min(single_rate)#单次最大收益和单次最大损失
    print('单次最大收益：',single_max_profit,'单次最大损失：',single_max_loss)
    win_rate_times=round(float(profit_rate_times)/transaction_times,4)#胜率
    print('胜率：',win_rate_times)
    
    avg_profit_rate_time=round(np.mean([j for j in single_rate if i>0]),4)#平均盈利率
    avg_loss_rate_time=round(np.mean([i for i in single_rate if i<=0]),4)#平均损失率
    avg_plt_rate=abs(round(avg_profit_rate_time/avg_loss_rate_time,2))#平均盈亏比
    print('平均盈利率（按次）：',avg_profit_rate_time,'平均亏损率（按次）：',avg_loss_rate_time,'平均盈亏比（按次）：',avg_plt_rate)

        
if __name__=='__main__':
    file_path='./data/000905.SH.mat'
    df=mat_to_df(file_path)
    df.date=df.date.apply(lambda i : str(int(i)))
    df3=df[618:].copy().reset_index()
    df1=slope_method1(df[600:],18)
    df1=df1.drop(['index'],axis=1)
#     plt.plot(df1['lowest_price_today'].values,df1['highest_price_today'])
#     plt.show()
#     df1.date=df1.date.apply(lambda i : str(int(i)))
#     print(df1.date)
#     df1.reset_index().to_csv('./data/000905.SH_1.csv',)
#     df1=slope_method1(df,18)
#     print(df1)
#     df1.day_slope=df1.day_slope.apply(lambda i:round(i,2))
#     pfr=pandas_profiling.ProfileReport(pd.DataFrame(df1.day_slope).reset_index())
#     pfr.to_file('report.html')#生成斜率报告
    '''
     RSRS 斜率指标交易策略为： 1. 计算 RSRS 斜率。 2. 如果斜率大于 1，则买入持有。 3. 如果斜率小于 0.75，则卖出手中持股平仓。 
     
    '''
    df2_=slop_method2(df,18,600)
    df2_=df2_.drop(['level_0','index'],axis=1)
#     print(df2_)
#     df2.z_score=df2.z_score.apply(lambda i:round(i,2))
#     pfr=pandas_profiling.ProfileReport(pd.DataFrame(df2.z_score).reset_index())
#     pfr.to_file('z_score_report.html')#生成标准分报告
    '''
            则 RSRS 标准分交易策略为： 1. 根据斜率计算标准分（参数 N=18,M=600）。 2. 如果标准分大于 S（参数 S=1），则买入持有。 3. 如果标准分小于-S，则卖出平仓。
    '''
#     exchange_detail1=apply_strategic1(df1,1,0.75)
    
#     base_net_value(df1)
#     net_value_slope_df=calcu_net_value(df1,method='slope')#计算slope净值
    net_value_z_score_df=calcu_net_value(df2_,method='z_score')#计算z_score净值
#     df3['base_value']=df3.closed_price_today/df1.closed_price_today.iloc[0]
    date=df3.date.values
#     print(len(net_value_slope_df['buy_sell']))
#     print(df3.base_value)
# #     print( net_value_z_score)
#     plot_net_value(df3.base_value,net_value_slope,net_value_z_score,date)
    statics_index(net_value_z_score_df)
    