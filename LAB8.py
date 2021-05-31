# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:36:30 2020

@author: ho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

os.chdir('D:/TEACHING/TEACHING KOREA/SEOULTECH/PYTHON/2020 SPRING/LABS/')

pd.date_range('1/1/2020', periods=5)
pd.date_range('1/1/2020', periods=5, freq='M')

# Umbrella Sales Example 우산세일 예제
# Create an umbrella time series (freq=‘Q’) and plot it using plt.plot. 4분기 시계열 데이터로 만든후 그래프로 그리시오. 
sales = [125,153,106,88,118,161,133,102,138,144,113,80,109,137,125,109,130,165,128,96]
t = pd.date_range('2015-01', periods=20, freq='Q')
s = pd.Series(sales, index=t)
plt.plot(s)
plt.ylim(0,200)
plt.title('Umbrella Sales')
plt.show()

# TV Sets Sales Example 티비세트 세일 예제
# Create a tvset time series (freq=‘Q’) and plot it using plt.plot.
# 4분기 시계열 데이터로 만든후 그래프로 그리시오.
sales = [4.8,4.1,6,6.5,5.8,5.2,6.8,7.4,6,5.6,7.5,7.8,6.3,5.9,8,8.4]
t = pd.date_range('2017-01', periods=16, freq='Q')
s = pd.Series(sales, index=t)
plt.plot(s)
plt.ylim(0,10)
plt.title('TV Sets Sales')
plt.xticks(rotation=60)
plt.show()

# Lawn-Maintenance Expense Example 잔디유지 비용 예제
# Create a lawn-maintenance time series (freq=‘M’) and plot it using plt.plot 월별 시계열 데이터로 만든후 그래프로 그리시오.
lawn = pd.read_excel('lawn.xlsx', header=None)
lawn.columns = ['rev']
lawn.index = pd.date_range('2018-01', periods=36, freq='M')
plt.plot(lawn)
plt.ylim(0,500)
plt.title('Lawn-Maintenance Expense')
plt.xticks(rotation=60)
plt.show()

# Import the bicycle data. 자전거 데이터를 가져오시오.
sales = [21.6,22.9,25.5,21.9,23.9,27.5,31.5,29.7,28.6,31.4]
t = np.arange(1,11)

# Plot the time series. 시계열 데이터 그래프 그리기
plt.plot(t, sales)
plt.ylim(0,35)
plt.title('Bicyle Sales')
plt.xlabel('Year')
plt.ylabel('Sales (1000s)')
plt.grid(axis='y', linestyle='--')
plt.show()

# Conduct a regression analysis and plot the regression equation on the time series plot. 회귀분석 및 그래프에 회귀식 추가
y = sales
x = t
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
#sales = 20.4 + 1.1 t

# Forecast year 11 and year 12. 11년, 12년 세일예측
test_data = [11, 12]
model.predict(sm.add_constant(test_data))

# Create an error table and calculate ME, MAE, MAPE, and MSE. 에러 테이블을 이용하여 예측에러 측정치들을 계산
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
pct_error = (abs_error / sales)*100
sq_error = error**2
me = round(error.mean(),2)
mae = round(abs_error.mean(),2)
mape = round(pct_error.mean(),2)
mse  = round(sq_error.mean(),2)
print(me, mae, mape, mse)

# Create a tracking signal table and plot the signals. 추적신호 테이블을 만들고 추적신호 그래프
sum_error = np.cumsum(error)
sum_ae = np.cumsum(abs_error)
mad = sum_ae / t
ts = sum_error / mad
plt.plot(ts)
plt.grid(axis='y', linestyle='--')
plt.title('Tracking Signal')
plt.show()

# Import the revenue data. 콜레스테롤 수익 데이터를 가져오기
rev = [23.1,21.3,27.4,34.6,33.8,43.2,59.5,64.4,74.2,99.3]
t = np.arange(1,11)
df = pd.DataFrame({'t':t, 'rev':rev})

# Plot the time series. 시계열 그래프 그리기
plt.plot(df.t, df.rev, marker='o')
plt.grid(axis='y', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Choresterol Revenue')
plt.show()

# Conduct a regression analysis and plot the regression equation on the time series plot. 회귀분석 수행과 회귀식 추가
df['t_sq'] = df.t**2
y = df.rev
x = df.drop('rev', axis=1)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
#revenue = 24.1817  - 2.1060 * t + 0.9216 * t^2

# Forecast year 11 and year 12. 11년, 12년 예측
test_data = pd.DataFrame({'t': [11, 12],
                           't_sq':[11**2, 12**2]})
model.predict(sm.add_constant(test_data))

# Create an error table and calculate ME, MAE, MAPE, and MSE. 에러 테이블과 에러측정치 계산
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
pct_error = (abs_error / sales)*100
sq_error = error**2
table = pd.concat([error,abs_error, pct_error, sq_error], axis=1)
table.columns = ['Error', 'Abs Error', '% Error', 'Sq Error']
table
me = round(error.mean(),2)
mae = round(abs_error.mean(),2)
mape = round(pct_error.mean(),2)
mse  = round(sq_error.mean(),2)
print(me, mae, mape, mse)

# Create a tracking signal table and plot the signals. 추석신호 테이블과 그래프
sum_error = np.cumsum(error)
sum_ae = np.cumsum(abs_error)
mad = sum_ae / t
ts = sum_error / mad
plt.plot(ts)
plt.grid(axis='y', linestyle='--')
plt.title('Tracking Signal')
plt.show()