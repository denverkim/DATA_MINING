# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:40:18 2020

@author: kim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

os.chdir('D:/TEACHING/TEACHING KOREA/SEOULTECH/PYTHON/2020 SPRING/LABS/')

# Umbrella Sales Example 우산세일 예제
umbrella = pd.read_excel('umbrella.xlsx')

# Plot the time series. 시계열데이터 그리기
umbrella.columns = ['Year', 'Quarter', 'Sales']
t = np.arange(1,21)
plt.plot(t, umbrella.Sales)
plt.ylim(0,200)
plt.show()

# Forecast year 6 using: 6년의 세일을 예측하시오. 
# Simple forecasting with seasonality 단순예측법 이용
umbrella.Sales[umbrella.Quarter == 1].mean()
umbrella.Sales[umbrella.Quarter == 2].mean()
umbrella.Sales[umbrella.Quarter == 3].mean()
umbrella.Sales[umbrella.Quarter == 4].mean()

# Multiple regression with seasonality  (dummy variables)  다중회귀분석 이용
umbrella['qrt1'] = [1,0,0,0]*5
umbrella['qrt2'] = [0,1,0,0]*5
umbrella['qrt3'] = [0,0,1,0]*5

y = umbrella.Sales
x = umbrella[['qrt1', 'qrt2', 'qrt3']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

#sales =  95 + 29 qrt1 + 57 qrt2 + 26 qrt3
95 + 29 * 1 + 57 * 0 + 26 * 0
95 + 29 * 0 + 57 * 1 + 26 * 0
95 + 29 * 0 + 57 * 0 + 26 * 1
95 + 29 * 0 + 57 * 0 + 26 * 0

test_t = pd.DataFrame({'qrt1':[1,0,0,0],
                       'qrt2':[0,1,0,0],
                       'qrt3':[0,0,1,0]})
test_t = sm.add_constant(test_t)
model.predict(test_t)

# Seasonal decomposition 계절성 분해법 이용
# Create an error table and calculate ME, MAE, MAPE, and MSE to compare three forecasting methods. Select the best method. 에러테이블, 측정치를 계산한후 두 모델을 비교해서 가장 좋은 모델을 선정하시오.
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
pct_error = abs_error/y
sq_error = error**2
me = error.mean()
mae = abs_error.mean()
mape = pct_error.mean()
mse = sq_error.mean()
me, mae, mape, mse

# TV Set Sales Example 티비세트 세일 예제
tv = pd.read_excel('tv.xlsx')

# Plot the time series. 시계열 데이터 그리기
t = np.arange(1,17)
plt.plot(t, tv.Sales)
plt.ylim(0,10)
plt.show()

# Forecast year 5. 5년을 예측하시오.
# Multiple regression with dummy variables 가변수를 이용한 회귀분석
tv['qrt1'] = [1,0,0,0] * 4
tv['qrt2'] = [0,1,0,0] * 4
tv['qrt3'] = [0,0,1,0] * 4
tv['t'] = t
y = tv.Sales
x = tv.drop(['Sales', 'Quarter', 'Year'], axis=1)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
#sales = 6.0688 - 1.3631*qrt1 - 2.0337*qrt2 - 0.3044*qrt3 + 0.1456*t
6.0688 - 1.3631*1 - 2.0337*0 - 0.3044*0 + 0.1456*17
6.0688 - 1.3631*0 - 2.0337*1 - 0.3044*0 + 0.1456*18
6.0688 - 1.3631*0 - 2.0337*0 - 0.3044*1 + 0.1456*19
6.0688 - 1.3631*0 - 2.0337*0 - 0.3044*0 + 0.1456*20

test_t = pd.DataFrame({'qrt1':[1,0,0,0],
                       'qrt2':[0,1,0,0],
                       'qrt3':[0,0,1,0],
                       't':[17,18,19,20]})
test_t = sm.add_constant(test_t)
model.predict(test_t)

# Seasonal decomposition 계절성 분해법
# Create an error table and calculate ME, MAE, MAPE, and MSE to compare two forecasting methods. Select the best method. 에러테이블, 측정치를 계산한후 두 모델을 비교해서 가장 좋은 모델을 선정하시오.
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
pct_error = abs_error/y
sq_error = error**2
me = error.mean()
mae = abs_error.mean()
mape = pct_error.mean()
mse = sq_error.mean()
me, mae, mape, mse