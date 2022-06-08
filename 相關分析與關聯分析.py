import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#%%H0:correlation=0  H1:cor!=0
car_corr=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch9/car_corr.csv')
#相關係數矩陣
np.corrcoef((car_corr['Weight'],car_corr['Circle'],car_corr['Max_Speed'],car_corr['Horsepower']))
car_corr.corr()
car_corr[['Weight','Circle','Max_Speed','Horsepower']].corr()
#相關係數檢定        H0:corr=0   H1:corr!=0
stats.pearsonr(car_corr['Max_Speed'],car_corr['Weight'])

correlation=[]
for i in car_corr[['Weight','Circle','Max_Speed','Horsepower']]:
    correlation.append(stats.pearsonr(car_corr['Max_Speed'],car_corr[i]))

from sklearn.feature_selection import f_regression
F,P_Value=f_regression(car_corr[['Weight','Circle','Max_Speed','Horsepower']],car_corr['Max_Speed'])
print(P_Value)
#%%偏相關係數分析(partial correlation analysis)    剔除重大影響變數  再分析相關係數
import pingouin as pg
for i in ['Weight','Circle']:
    print(pg.partial_corr(data=car_corr,x=i,y='Max_Speed',covar='Horsepower'))
#%%點二列相關分析(point-biserial correlation)    數值型(常態分布)與二元分類資料相關  H0:correlation=0  H1:cor!=0
scorebygender=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch9/scorebygender.csv')
stats.pointbiserialr(scorebygender['gender'],scorebygender['score'])
#p>0.05  不顯著  成績與性別無關
#%%非參數相關分析:次序間的關係
#spearman相關係數
#kendall tau-b係數
#Hoeffding's D係數
graduate=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/ch9/graduate.csv')
rho,p=stats.spearmanr(graduate)
print(rho)#'相關係數:'
print(p)#'p-value:'

kt=[]
for i in graduate.keys():
    kt.append(stats.kendalltau(graduate[i],graduate['Tutor']))


#%%
market=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/數據分析基礎/Python数据分析基础（第2版）数据/Market_Basket_Optimisation.csv')
# 資料及放到list中
transacts = []
for i in range(0, len(market)): 
  transacts.append([str(market.values[i,j]) for j in range(0, 20)])
#apriori算法
from apyori import apriori
rule = apriori(transactions = transacts, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
output = list(rule) # returns a non-tabular output
# putting output into a pandas dataframe
def inspect(output):
    lhs         = [tuple(result[2][0][0])[0] for result in output]
    rhs         = [tuple(result[2][0][1])[0] for result in output]
    support    = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift       = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))
output_DataFrame = pd.DataFrame(inspect(output), columns = ['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])
print(output_DataFrame)
print(output_DataFrame.nlargest(n = 10, columns = 'Lift'))

































