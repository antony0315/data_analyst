import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pylab as plt
%matplotlib inline
#%%一元線性回歸
murder=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch10/murder.csv')
murder['region']=murder['region'].astype('category')
murder['region'].cat.categories=['East North Central','East South Central',
                                 'Middle Atlantic','Mountain',
                                 'New England','Pacific','South Atlantic',
                                 'West North Central','West South Central']
plt.scatter(murder['illiteracy'],murder['murder'])
#model
murder_model1=stats.linregress(murder['illiteracy'],murder['murder'])
print(murder_model1)
print('r_square:%.5f'%murder_model1.rvalue**2)
            #murder=2.396775611095853+4.257*(illiteracy)

#方法2
from statsmodels.formula.api import ols
formula='murder~illiteracy'
murder_model2=ols(formula,data=murder).fit()
murder_model2.summary2()

#QQ圖診斷殘差符合常態分佈
sm.qqplot(murder_model2.resid,fit=True,line='45')
sm.ProbPlot(murder_model2.resid,fit=True).ppplot(line='45')
sm.ProbPlot(murder_model2.resid,fit=True).qqplot(line='45')
#殘插圖(2*2)
fig=plt.figure(figsize=(12,8))
from statsmodels.graphics.regressionplots import plot_regress_exog
plot_regress_exog(murder_model2,1,fig=fig)
#置信區間
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd,interval_l,interval_u=wls_prediction_std(murder_model2,alpha=0.05)#95%信賴區間
fig=plt.subplots(figsize=(7,4))
plt.plot(murder['illiteracy'],murder['murder'],'o',label="data")
plt.plot(murder['illiteracy'],murder_model2.fittedvalues,'r--',label="ols")
plt.plot(murder['illiteracy'],interval_u,'r--')
plt.plot(murder['illiteracy'],interval_l,'r--')
plt.legend(loc='best')




#%% 多元線性回歸
from statsmodels.formula.api import ols
salary=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch10/salary_r.csv')
salary=salary.dropna(axis=0)
salary.info()
salary['position']=salary['position'].astype('category')
salary['position'].cat.categories=['經理','主管','普通員工']
salary['Gender']=salary['Gender'].astype('category')
salary['Gender'].cat.categories=['女','男']
formula='Current_Salary~Education+Begin_Salary+Experience+Age'
salary_model1=ols(formula,data=salary).fit()
salary_model1.summary2()
#P-value >0.1不顯著   剔出模型
formula='Current_Salary~Education+Begin_Salary+Experience'
salary_model2=ols(formula,data=salary).fit()
salary_model2.summary2()
#
fig=plt.gcf()
fig.suptitle('Residual by regression for salary')
fig.set_size_inches(6,6)
plt.subplot(221)
plt.plot(salary['Education'],salary_model2.resid,'o')
plt.xlabel('Education')
plt.ylabel('Residual')
plt.subplot(222)
plt.plot(salary['Begin_Salary'],salary_model2.resid,'o')
plt.xlabel('Begin_salary')
plt.subplot(223)
plt.plot(salary['Experience'],salary_model2.resid,'o')
plt.xlabel('Experience')
plt.ylabel('Residual')
plt.subplot(224)
plt.plot(salary_model2.fittedvalues,salary_model2.resid,'o')
plt.xlabel('predict')

#%%虛擬變數回歸
from patsy.contrasts import Treatment
contrast=Treatment(reference=3).code_without_intercept([1,2,3])
print(contrast)
from statsmodels.formula.api import ols
formula='Current_Salary~Education+Begin_Salary+Experience+Age+C(position)+C(Gender)'
salary_model4=ols(formula,data=salary).fit()
print(salary_model4.summary2())
#%%非線性回歸
eb=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch10/electronic_business.csv')
plt.scatter(eb['registration'],eb['sales'],marker='o')
plt.xlabel('registration')
plt.ylabel('sales')
formula='sales~registration+np.square(registration)'
eb_model=ols(formula,data=eb).fit()
print(eb_model.summary2())
#視覺化
sm.graphics.plot_fit(eb_model,1)
plt.show()



#%%===擬合
eb_ex=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch10/eb_extended.csv')
plt.scatter(eb_ex['registration'],eb_ex['sales'],marker='o')
from scipy.optimize import curve_fit
#觀察x<=20為拋物線   x>20為指數      <=20:a1+b*r1+b**2*r2+e    >20:a2*exp**(reg)+e
def func1(x,alpha2,beta3,gamma):
    return alpha2*(gamma**(beta3*x))
def func2(x,alpha1,beta1,beta2):
    return alpha1+beta1*x+beta2*np.square(x)
m1=curve_fit(func1,eb_ex[eb_ex['registration']>20]['registration'],eb_ex[eb_ex['registration']>20]['sales'])
m2=curve_fit(func2,eb_ex[eb_ex['registration']<=20]['registration'],eb_ex[eb_ex['registration']<=20]['sales'])
m1,m2

eb_ex.loc[eb_ex['registration']>20,['predicted']]=func1(eb_ex[eb_ex['registration']>20]['registration'],m1[0][0],m1[0][1],m1[0][2])
eb_ex.loc[eb_ex['registration']<=20,['predicted']]=func2(eb_ex[eb_ex['registration']<=20]['registration'],m2[0][0],m2[0][1],m2[0][2])

eb_ex_sorted=eb_ex.sort_values(by='registration')
plt.plot(eb_ex_sorted['registration'],eb_ex_sorted['sales'],'o',
         eb_ex_sorted['registration'],eb_ex_sorted['predicted'],'r--',lw=3)

#%%多項式回歸
from statsmodels.formula.api import ols
lorenz=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch10/lorenz.csv')
lorenz_est=np.polyfit(lorenz['cpop'],lorenz['cincome'],2)
lorenz_est
lorenz_poly=np.poly1d(lorenz_est)
print(lorenz_poly)


formula='cincome~cpop+np.square(cpop)'
lorenz_model=ols(formula,data=lorenz).fit()
lorenz_model.summary2()
sm.graphics.plot_fit(lorenz_model,1)
plt.show()

def cube(x):
    return x**3
formula2='cincome~cpop+np.square(cpop)+cube(cpop)'
lorenz_model2=ols(formula2,data=lorenz).fit()
lorenz_model2.summary2()
sm.graphics.plot_fit(lorenz_model2,1)
plt.show()




#%%分位數回歸
from statsmodels.formula.api import quantreg
from statsmodels.formula.api import ols
formula='Current_Salary~Begin_Salary+Education+Experience'
salary_qt=quantreg(formula,data=salary)
salary_qtmodel1=salary_qt.fit(q=0.1)
print(salary_qtmodel1.summary())



















































