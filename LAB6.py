# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:00:00 2020

@author: kim
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
os.chdir('E:/TEACHING/TEACHING KOREA/SEOULTECH/PYTHON/2020 SPRING/LABS/')

# Create an Excel file with butler data and read it into a data frame 버틀러 데이터를 엑셀파일로 만들어서 데이터프레임으로 읽으세요.
butler = pd.read_excel('butler.xlsx')

# Draw the scatter plot 산점도 그래프를 그리시요.
butler.columns
x = butler.MilesTraveled
y = butler.TravelTime
plt.scatter(x,y)
plt.xlabel('Miles Traveled')
plt.ylabel('Travel Time')
plt.title('Scatter Plot')
plt.xlim(0, 120)
plt.ylim(0, 10)
plt.grid()
plt.show()

# Develop the estimated simple linear regression equation to predict travel time given miles traveled. Does the equation that you developed provide a good fit for the observed data? Explain. 운전거리를 이용하여 운전시간을 예측하는 선형회귀식을 구하시요. 예측식은 주어진 데이터를 잘 나타내고 있습니까?
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

# Develop the estimated regression equation to predict travel time given the number of deliveries and miles traveled. At the .05 level of significance, test whether the estimated regression equation represents a significant relationship between the independent variables and dependent variable. 배달수와 운전거리를 가지고 운전시간을 예측하는 회귀식을 개발하시요. 독립변수와 종속변수사이에 관계가 있는지 예측식을 0.05 유의수준에서 테스트하시요. 
x = butler[['MilesTraveled', 'NumberOfDeliveries']]
y = butler.TravelTime
x = sm.add_constant(x)
model1 = sm.OLS(y, x).fit()
model1.summary()
# Is miles traveled statistically significant? Use alpha = .05. What explanation can you give for the results observed? 0.05유의수준에서 운전거리가 통계적으로 유의한지 설명하시요.
# Is number of deliveries statistically significant? Use alpha = .05. What explanation can you give for the results observed? 0.05수준에서 배달수가 통계적으로 유의한지 설명하시요.

# Calculate the residual. 잔차를 계산하시요.
y_pred = model1.predict(x)
residual = y - y_pred
# Draw the residual plot. 잔차도를 그리시요.
# Residual plot against Miles Traveled. 이동거리에 대한 잔차도
plt.scatter(butler.MilesTraveled, residual)
plt.xlabel('Miles Traveled')
plt.ylabel('Residual')
plt.title('Residual Plot againt Miles Traveled')
plt.grid()
plt.xlim(0,120)
plt.ylim(-2, 2)
plt.show()

# Residual plot against Number of Deliveries. 배달수에 대한 잔차도
plt.scatter(butler.NumberOfDeliveries, residual)
plt.xlabel('Number Of Deliveries')
plt.ylabel('Residual')
plt.title('Residual Plot againt Number Of Deliveries')
plt.grid()
plt.xlim(0,6)
plt.ylim(-2, 2)
plt.show()

# Residual plot against Predicted Travel Time. 예측운전시간에 대한 잔차도
plt.scatter(y_pred, residual)
plt.xlabel('Predicted Travel Time')
plt.ylabel('Residual')
plt.title('Residual Plot againt Predicted Travel Time')
plt.grid()
plt.xlim(0,15)
plt.ylim(-2, 2)
plt.show()

# Calculate the standard residual. 표준잔차를 계산하시요.
std_residual = residual / np.std(residual)

# Draw the standard residual plot against Predicted Travel Time. 예측운전시간에 대한 표준잔차도를 그리시요.
plt.scatter(y_pred, std_residual)
plt.xlabel('Predicted Travel Time')
plt.ylabel('Standard Residual')
plt.title('Standard Residual Plot againt Predicted Travel Time')
plt.grid()
plt.xlim(0,15)
plt.ylim(-2, 2)
plt.show()