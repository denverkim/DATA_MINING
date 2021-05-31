# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:56:42 2020

@author: Kim
"""

import pandas as pd
import numpy as np

#씨리즈
s = pd.Series([3,5,4,3], index=['a', 'b', 'c', 'd'])

#데이터프레임
data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasilia'],
        'Population': [3345566, 33322, 455666]}
df = pd.DataFrame(data)
df

data1 = [['Jan', 34, 45, 33, 33, 3.45],
         ['Feb', 55, 33, 67, 88, 3.21],
         ['Mar', 44, 22, 66, 43, 5.43],
         ['Apr', 55, 66, 33, 44, 2.34]]
df1 = pd.DataFrame(data1, columns=['month', 'avg_high', 
                             'avg_low', 'record_high', 
                             'record_low', 'avg_precipitation'])
df1

#클래스 데이터프레임 예제
data2 = {'courseID': [1111, 2222, 3333],
         'courseName':['math', 'english', 'science'],
         'creditHours':[3,2,3],
         'Days':['mon', 'tue', 'wed']}
df2 = pd.DataFrame(data2)
df2 
df2.to_excel('D:\\TEACHING\\TEACHING KOREA\\SEOULTECH\\PYTHON\\2020 SPRING\\LABS\\courses.xlsx')
import os
os.getcwd()
os.chdir('D:\\TEACHING\\TEACHING KOREA\\SEOULTECH\\PYTHON\\2020 SPRING\\LABS\\')

# Read weather.txt file into a dataframe. 날씨파일을 데이터프레임으로 읽기
weather = pd.read_csv('weather.txt')
# Print first 5 or last 3 rows of df 처음 다섯줄, 마지막 세줄 출력
weather.head()
weather.tail(3)
# Get data types, index, columns, values 데이터타입, 인덱스, 컬럼들, 값들
weather.dtypes
weather.index
weather.columns
weather.values
# Statistical summary of each column 각 컬럼에 대한 통계요약
a = weather.describe()
# Sort records by any column (descending order) 한 컬럼을 정해 내림차순으로 정렬 
weather.sort_values(by='avg_precipitation',
                    ascending=False)

# Slice the records and display the following columns and rows:
# 레코드를 짤라서 다음의 열과 행을 출력
# avg_low 
weather.avg_low
# Rows 1 to 2
weather[0:2]
weather.iloc[0:2]
# avg_low and avg_high
weather[['avg_low', 'avg_high']]
# 9 row of avg_precipitation column
weather.avg_precipitation[8]
weather.loc[8,'avg_precipitation']
# 4 to 5 rows of 1 and 4 columns
weather.iloc[3:5,[0,3]]

# Filter the data and display the following columns and rows: 데이터를 검색하여 다음의 열과 행을 출력
# avg_precipitation > 1.0
weather[weather.avg_precipitation > 1]
# Month is in either June, July, or August
weather[weather.month.isin(['Jun', 'Jul', 'Aug'])]

# Assign new values in the following locations: 새로운 값으로 대체
# 101.3 for avg_precipitation column at index 9 
weather.avg_precipitation[9] = 101.3
weather.avg_precipitation
# Np null values for avg_precipitation column at index 9 (np.nan)
weather.loc[9, 'avg_precipitation'] = np.nan
# 5 for all rows in avg_low column avg_low 컬럼 전체를 5로 바꾸기
weather.avg_low = 5
# Add new column named avg_day that is the average of avg_low and avg_high avg_low와 avg_high의 평균을 구하여 avg_day컬럼을 만듬
weather['avg_day'] = (weather.avg_high + weather.avg_low)/2

# Rename columns 컬럼이름 바꾸기
# avg_precipitation to avg_rain
weather.rename(columns={'avg_precipitation':'avg_rain'},
               inplace=True)
# Change columns’ name to 'month','av_hi','av_lo','rec_hi','rec_lo','av_rain‘, ‘av_day’
weather.columns = ['month', 'av_hi', 'av_lo', 'rec_hi',
                   'rec_lo', 'av_rain', 'av_day']
# Save the result data frame to a csv file 데이터프레임을 csv파일로 저장
weather.to_csv('weather.csv')