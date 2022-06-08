import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pylab as plt
%matplotlib inline
from statsmodels.formula.api import ols
#%%二元回歸   Binary Probit
from statsmodels.formula.api import glm
product=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch11/product_usage.csv')
formula='Attitude~CSI+Complaint+Loyalty'
product_m=glm(formula,data=product,family=sm.families.Binomial(link=sm.families.links.probit())).fit()
print(product_m.summary())
product_m.predict(pd.DataFrame({'CSI':[8],'Complaint':[4],'Loyalty':[7]}))

#===類別變數  二元回歸  Binary probit regression
G3=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch11/threeg.csv')
G3['Gender']=G3['Gender'].astype('category')
G3['Gender'].cat.categories=['女','男']
formula='Purchase~C(Gender)+Age'
G3_model1=glm(formula,data=G3,family=sm.families.Binomial(sm.families.links.probit())).fit()
print(G3_model1.summary())
G3_model1.predict(pd.DataFrame({'Gender':['男','女'],'Age':[45,45]}))
#===方法2
from statsmodels.formula.api import probit
G3_model2=probit(formula,data=G3).fit()
print(G3_model2.summary2())
print(G3_model2.get_margeff().summary())  #邊際影響

#%%Binary_Logit_model
G3_model3=glm(formula,data=G3,family=sm.families.Binomial(sm.families.links.logit())).fit()
print(G3_model3.summary())
G3_model3.predict(pd.DataFrame({'Gender':['男','女'],'Age':[45,45]}))
#方法2
from statsmodels.formula.api import logit
G3_model4=logit(formula,data=G3).fit()
print(G3_model4.summary2())
print(G3_model4.get_margeff().summary())

#%%多重選擇模型
G3_m=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch11/threeg_multi.csv')
G3_m['gender']=G3_m['gender'].astype('category')
G3_m['gender'].cat.categories=['女','男']
# G3_m['purchase']=G3_m['purchase'].astype('category')
# G3_m['purchase'].cat.set_categories=['不買','無','買']

from statsmodels.formula.api import mnlogit
formula='purchase~C(gender)+age'
G3_m_model=mnlogit(formula,data=G3_m).fit()
print(G3_m_model.summary())
G3_m_model.predict(pd.DataFrame({'gender':['男','女'],'age':[45,45]}))

#%%計數模型  Poisson Regression(疫苗保護力)
printer=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch11/printer.csv')
printer['cartridge']=printer['cartridge'].astype('category')
printer['cartridge'].cat.categories=['原廠','相容']
from statsmodels.formula.api import poisson
formula='Counts~C(cartridge)+Pages+Length'
printer_model=poisson(formula,data=printer).fit()
print(printer_model.summary2())
print('預測發生的次數',printer_model.predict(pd.DataFrame({'cartridge':['原廠'],'Pages':[92],'Length':[46.228]})))





