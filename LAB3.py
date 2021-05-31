# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:49:54 2020

@author: Kim
"""

import os
os.chdir('D:\\TEACHING\\TEACHING KOREA\\SEOULTECH\\PYTHON\\2020 SPRING\\LABS')
import pandas as pd
# Import the en.openfoodfacts.org.products.tsv dataset and assign it to a dataframe called food. 세계음식팩트 파일을 임포트해서 food라는 데이터프레임에 넣으시요.
food = pd.read_csv('worldfoodfact.tsv', delimiter='\t')
# Display the first 5 rows 처음 5줄
food.head()
# What is the number of observations in the dataset? 관측치의 갯수
food.shape[0]
len(food)
# What is the number of columns in the dataset? 컬럼의 수
len(food.columns)
food.columns.size
# Print the name of all the columns. 모든 컬럼의 이름
food.columns
# What is the name of 105th column? 105번째 컬럼의 이름
food.columns[104]
# What is the data type of the observations of the 105th column? 105번째 컬럼의 데이터타입
food.dtypes[104]
# How is the dataset indexed? 데이터셋은 어떻게 인덱스되었나요?
food.index
# What is the product name of the 19th observation? 19번째 관측치의 상품이름은?
food.product_name[18]
# Import the chipo dataset and assign it to a variable called chipo.
# 데이터셋을 임포트해서 chipo라고 하시요
chipo = pd.read_excel('chipotle.xlsx')
# Display the first 10 rows 처음 10줄
chipo.head(10)
# What is the number of observations in the dataset? 관측치의 갯수는?
len(chipo)
# What is the number of columns in the dataset? 컬럼의 수는?
len(chipo.columns)
# Print the name of all the columns. 모든 컬럼의 이름을 출력
chipo.columns
# How is the dataset indexed? 데이터셋은 어떻게 인덱스 되었나요?
chipo.index
# How many items were ordered in total? 
# 전체 주문 아이템수는? Quantity의 합
chipo.quantity.sum()
# How much was the revenue for the period in the dataset? 
# 데이터셋안에 있는 기간에 대한 수익은 얼마인가요?
(chipo.quantity * chipo.item_price).sum()
# Which was the most-ordered item? 가장 많이 주문된 아이템은? 
#Item_name에 따라 groupby된 것중 가장 카운트가 많이 된 것을 찾음
a = chipo.groupby('item_name').count().sort_values(by='order_id', ascending=False)
a.index[0]
# For the most-ordered item, how many items were ordered? 
# 가장 많이 주문된 아이템은 몇개나 주문됐나요? Groupby된것의 카운트수
a.order_id[0]
# What was the most ordered item in the choice_description column? 
#choice_description컬럼에서 가장 많이 주문된 아이템은? choice_description에 따라 groupby된 것중 가장 카운트가 많은 것
b = chipo.groupby('choice_description').count().sort_values(by='order_id', ascending=False)
b.index[0]
# How many different items are sold? 얼마나 많은 다른 종류의 아이템이 팔렸나요?
chipo.item_name.nunique()
# Import the occupation dataset and assign it to a variable called users and use the 'user_id' as index 데이터셋을 임포트해서 users라는 변수에 저장한후 user_id를 인덱스로 사용하시요
users = pd.read_csv('occupation.txt', delimiter='|')
# Display the first 25 rows 처음 25줄
users.head(25)
# Display the last 10 rows 마지막 10줄
users.tail(10)
# What is the number of observations in the dataset? 관측치의 수
len(users)
# What is the number of columns in the dataset? 컬럼의 수
users.columns.size
# Print the name of all the columns. 모든 컬럼의 이름
users.columns
# How is the dataset indexed? 데이터셋은 어떻게 인덱스 되었나요?
users.index
# What is the data type of each column? 각 컬럼의 데이터타입은?
users.dtypes
# Print only the occupation column 직업컬럼만 출력
users.occupation
# Summarize the data frame (descriptive statistics) 기술통계를 이용하여 데이터요약
users.describe()
# Summarize all the columns 모든 컬럼을 요약
users.describe(include='all')
# Summarize only the occupation column 직업 컬럼만 요약
users.occupation.describe()
# What is the mean age of users? 모든 사용자의 나이의 평균
users.age.mean()
users['age'].mean()
# How many different occupations there are in this dataset? 직업의 종류는?
users.occupation.nunique()
# What is the most frequent occupation? 가장 많은 직업은?
users.occupation.value_counts().index[0]
# What is the age with least occurrence? 가장 적게 나오는 나이는?
users.age.value_counts(ascending=True).head()
users.age.value_counts().tail()

import matplotlib.pyplot as plt
#간단한 선그래프
plt.plot([1,2,3], [5,7,4])
plt.show()

x = [1, 2, 3]
y = [5, 5, 8]
y2 = [10, 15, 6]
plt.plot(x, y, label='first line')
plt.plot(x, y2, label='second line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Chart')
plt.show()

#산점도
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [3, 5, 6, 8, 3, 8, 9, 3]
plt.scatter(x,y, color='k', s=100, marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot\n Example')
plt.show()

#히스토그램
ages = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 99, 102, 110, 120, 121, 122, 130, 111, 115, 112, 80,75, 65, 54, 44, 43, 42, 48]
bins = list(range(0,130,10))
plt.hist(ages, bins, rwidth=.8)
plt.xlabel('ages')
plt.ylabel('count')
plt.title('Histogram')
plt.show()

#막대그래프
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 2, 4]
plt.bar(x,y, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Chart')
plt.show()

#수평막대그래프
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 2, 4]
plt.barh(x,y, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Chart')
plt.show()
 
#combined chart
x2 = [1, 3, 5, 7, 9]
y = [6, 7, 8, 2, 4]
x = [2, 4, 6, 8, 10]
y2 = [7, 8, 2, 4, 2]
plt.bar(x, y, color='r', label='bar1')
plt.bar(x2, y2, color='c', label='bar2')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Two Bar Charts')
plt.show()

#누적그래프
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 8, 2, 4]
y2 = [7, 8, 2, 4, 2]
plt.bar(x, y1, color='r', label='bar1')
plt.bar(x, y2, bottom=y1, color='c', label='bar2')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stacked Bar Chart')
plt.show()

#박스플랏
data = [[2, 4, 6, 8, 10],
        [6, 7, 8, 2, 4],
        [1, 3, 5, 7, 9],
        [7, 8, 2, 4, 2]]
x = pd.DataFrame(data)
plt.boxplot(x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Boxplot')
plt.show()

#영역그래프
days = [1, 2, 3, 4, 5]
sleeping = [7, 8, 6, 11, 7]
eating = [2, 3, 4, 3, 2]
working = [7, 8, 7, 2, 2]
playing = [8, 5, 7, 8, 13]
plt.stackplot(days, sleeping, eating, working, playing,
              colors=['m', 'c', 'r','k'])
plt.legend(labels=['sleeping', 'eating', 'working','playing'])

#heatmap
data=[[2,3,4,1],[6,3,5,2],[6,3,5,4],[3,7,5,4],[2,8,1,5]] #5x4 matrix
data
df = pd.DataFrame(data, columns=['c1', 'c2','c3', 'c4'])
plt.pcolor(df.corr())
plt.matshow(df.corr())
plt.colorbar()

#pie chart
slices = [7, 2, 2, 13]
activity = ['sleeping', 'eating', 'working', 'playing']
cols = ['c', 'm', 'r', 'k']
plt.pie(slices, labels=activity, colors=cols,
        startangle=90, shadow=True, 
        explode=(0, 0.1, 0, 0),
        autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()