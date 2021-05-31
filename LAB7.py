# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 10:31:29 2020

@author: ho
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('z:/TEACHING/TEACHING KOREA/SEOULTECH/PYTHON/2020 SPRING/LABS/')
cravens = pd.read_excel('cravens.xlsx')

# Descriptive statistics 기술통계
# Summary statistics for all variables (describe) 모든 변수에 대한 기초통계
a = cravens.describe()
# Summary statistics (mean, median, mode, variance, standard deviation, coefficient of variance, and skewness) for Sales 총판매량에 대한 통계
cravens.columns
cravens.Sales.mean()
cravens.Sales.median()
cravens.Sales.mode() #최빈수
cravens.Sales.value_counts()
cravens.Sales.var()
cravens.Sales.std()
(cravens.Sales.std()/cravens.Sales.mean())*100 #변동계수
cravens.Sales.skew()

# Univariate analysis 일변량분석
# Frequency, percent frequency, and histogram using bar chart 
# 빈도표, 백분율빈도표, 히스토그램
sales = cravens.Sales
(sales.max() - sales.min())/9 #500
breaks = np.arange(1500, 7500, 500)
sales_cut = pd.cut(sales, breaks, include_lowest=True)
freq = sales_cut.value_counts(sort=False)
r_freq = freq/len(sales)
p_freq = r_freq * 100
freq_table = pd.concat([freq, r_freq, p_freq], axis=1)
freq_table.columns = ['Freq', 'Rel Freq', '% Freq']
sales.hist(bin=breaks)
cravens.hist()
plt.tight_layout()
plt.show()
cravens.boxplot()

# Multivariate analysis 다변량분석
# Correlation matrix, scatter plot (pairplot)
# 상관관계, 산점도
b = cravens.corr()
np.corrcoef(cravens)

from pandas.plotting import scatter_matrix
scatter_matrix(cravens)
%matplotlib auto
import seaborn as sns
sns.pairplot(cravens)

# Multiple regression with: 회귀분석
#  all variables 모든 변수사용
import statsmodels.api as sm
x = cravens.drop('Sales', axis=1)
y = cravens.Sales
x = sm.add_constant(x)
model1 = sm.OLS(y,x).fit()
model1.summary()
#r squared =  0.922

# 'Poten', 'AdvExp', 'Share‘
x = cravens[['Poten', 'AdvExp', 'Share']]
y = cravens.Sales
x = sm.add_constant(x)
model2 = sm.OLS(y,x).fit()
model2.summary()
#r squared = 0.849

# 'Poten', 'AdvExp', 'Share', 'Accounts‘
x = cravens[['Poten', 'AdvExp', 'Share', 'Accounts']]
y = cravens.Sales
x = sm.add_constant(x)
model3 = sm.OLS(y,x).fit()
model3.summary()
#r squared = 0.900

# 'Share', 'Change', 'Accounts', 'Work', 'Rating‘
x = cravens[['Share', 'Change', 'Accounts', 'Work', 'Rating']]
y = cravens.Sales
x = sm.add_constant(x)
model4 = sm.OLS(y,x).fit()
model4.summary()
#r squared = 0.700
# 'Poten', 'AdvExp', 'Share', 'Change', 'Time‘
x = cravens[['Poten', 'AdvExp', 'Share', 'Change', 'Time']]
y = cravens.Sales
x = sm.add_constant(x)
model5 = sm.OLS(y,x).fit()
model5.summary()
#r squared = 0.915

# Compare above models and determine a best model. 
# 위 모델들을 비교하고 가장 좋은 모델을 선택
#3번모델
# Sales = 0.0382 * Poten + 0.1750 * AdvExp + 190.1442* Share + 9.2139 * Accounts - 1441.9323

# Standard residual plot for the best model. 가장 좋은 모델의 표준잔차플랏
y_pred = model3.predict(x)
residual = y - y_pred
std_residual = residual / residual.std()
plt.scatter(y_pred, std_residual)
plt.grid()
plt.xlabel('Predicted Sales')
plt.ylabel('Standard Residual')
plt.title('Standard Residual Plot Against Predicted Sales')
plt.show()

# Prediction with test data (Poten=74065.1, AdvExp=4582.9, Share=2.51, Accounts:74.86). 테스트데이터로 예측
x.columns
test_data = pd.DataFrame({'const':1,
              'Poten':74065.1,
              'AdvExp':4582.9, 
              'Share':2.51, 
              'Accounts':74.86}, index=[0])
model3.predict(test_data)