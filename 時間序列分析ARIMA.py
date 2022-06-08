import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%%
sales=pd.read_sas('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch17/sales_monthly.sas7bdat')
sales.index=pd.Index(pd.date_range('1/2001','9/2008',freq='1M'))


#%%平穩性檢驗
sales['Sales'].plot()
sales['Sales'].diff(1).plot()
from statsmodels.tsa.stattools import acf,pacf
ts_d1_ACF=pd.DataFrame(acf(sales['Sales'].diff(1).iloc[1:92]),columns=['ACF'])
ts_d1_ACF['PACF']=pd.DataFrame(pacf(sales['Sales'].diff(1).iloc[1:92]))
import statsmodels.api as sm
fig=plt.figure(figsize=(10,6))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(sales['Sales'].diff(1).iloc[1:92],lags=24,ax=ax1)
#平穩性檢驗    H0:平穩白色雜訊檢驗結果不顯著   H1:不平穩，雜訊檢驗顯著  a=0.05
r,q,p=sm.tsa.acf(sales['Sales'].diff(1).iloc[1:92].values.squeeze(),qstat=True)
mat=np.c_[range(1,20),r[1:],q,p]
table=pd.DataFrame(mat,columns=['lag','AC','Q','porb(>Q)'])
LB_result=table.loc[[5,11,17]]
LB_result.set_index('lag',inplace=True)
print(LB_result)   
#單位跟檢驗(更精確檢定平穩性)   H0:時間序列有單位根(不平穩)   H1:無單位根(平穩)
from statsmodels.tsa.stattools import adfuller
def DFTest(sales,regression,maxlag,autolag='AIC'):
    print("ADF-Test Result")
    dftest=adfuller(sales,regression=regression,
                    maxlag=maxlag,autolag=autolag)
    dfoutput=pd.Series(dftest[0:2],index=['Test Statistic','p-value'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value at %s'%key]=value
    print(dfoutput)
DFTest(sales['Sales'],regression='nc',maxlag=6,autolag='AIC')
print(67*'-')
DFTest(sales['Sales'].diff(1).iloc[1:92],regression='nc',maxlag=5,autolag='AIC')
#%%
#ACF PAC圖模型識別
fig=plt.figure(figsize=(10,6))
ax1=fig.add_subplot(221)
fig=sm.graphics.tsa.plot_acf(sales['Sales'].diff(1).iloc[1:92].dropna(),lags=24,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(sales['Sales'].diff(1).iloc[1:92].dropna(),lags=24,ax=ax2)

#最小資訊準則模型識別
from statsmodels.tsa.arima_model import ARMAResults 
order_p,order_q,bic=[],[],[]
model_order=pd.DataFrame()
for p in range(4):
    for q in range(4):
        arma_model=sm.tsa.ARIMA(sales['Sales'].diff(1).iloc[1:92].dropna(),order=(p,0,q)).fit()
        order_p.append(p)
        order_q.append(q)
        bic.append(arma_model.bic)
        print('BIC of ARMA(%s,%s)is%s'%(p,q,arma_model.bic))

model_order['p']=order_p
model_order['q']=order_q
model_order['BIC']=bic
P=list(model_order['p'][model_order['BIC']==model_order['BIC'].min()])
Q=list(model_order['q'][model_order['BIC']==model_order['BIC'].min()])
print('ARMA最佳模型(%s,%s)參數'%(P[0],Q[0]))

#模型參數估計檢驗
model=sm.tsa.ARIMA(sales['Sales'].diff(1).iloc[1:92].dropna(),order=(0,0,2)).fit()
params=model.params
tvalues=model.tvalues
pvalues=model.pvalues
result_mat=pd.DataFrame({'Estimate':params,'t-values':tvalues,'p-values':pvalues})
print(result_mat)

resid=model.resid
r,q,p=sm.tsa.acf(resid.values.squeeze(),qstat=True)
mat_res=np.c_[range(1,20),r[1:],q,p]
table_res=pd.DataFrame(mat_res,columns=['to lag','AC','Q','Prob(>Q)'])
LB_result_res=table_res.loc[[5,11,17,18]]
LB_result_res.set_index('to lag',inplace=True)
print('殘差白色噪音結果',LB_result_res)
#ACF圖
from scipy import stats
fig=plt.figure(figsize=(10,6))
ax3=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(resid,lags=24,ax=ax3)
sm.ProbPlot(resid,stats.t,fit=True).ppplot(line='45')
sm.ProbPlot(resid,stats.t,fit=True).qqplot(line='45')
plt.show()
plt.figure()
x=pd.Series(resid)
p1=x.plot(kind='kde')
p2=x.hist(density=True)
plt.grid(True)
plt.show()

#%%預測
arma_model.forecast(steps=4)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.plot(sales['Sales'].diff(1).iloc[1:92],color='blue',label='Sales')
ax.plot(arma_model.fittedvalues,color='green',label='predicted sales')
plt.legend(loc='lower right')


#%%預測值還原(經過一階插分)
def forecast(step,var,modelname):
    diff=list(modelname.predict(len(var)-1,len(var)-1+step,dynamic=True))
    prediction=[]
    prediction.append(var[len(var)-1])
    seq=[]
    seq.append(var[len(var)-1])
    seq.extend(diff)
    for i in range(step):
        v=prediction[i]+seq[i+1]
        prediction.append(v)
    prediction=pd.DataFrame({'predicted':prediction})
    return prediction[1:]
print(forecast(4,sales['Sales'],model))











