# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:38:39 2020

@author: ho
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

#히스토그램
ages = [22, 55, 62, 45, 21, 22, 34, 42, 42, 4, 99, 102, 110, 120, 121, 122, 130, 111, 115, 112, 80,75, 65, 54, 44, 43, 42, 48]
bins = list(range(0,130,10))
plt.hist(ages, bins, rwidth=.8)
plt.xlabel('ages')
plt.ylabel('count')
plt.title('Histogram')
plt.show()

#누적그래프
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 8, 2, 4]
y2 = [7, 8, 2, 4, 2]
plt.bar(x,y1, label='y1', color='r')
plt.bar(x,y2, label='y2', color='c', bottom=y1)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stacked Bar Chart')
plt.show()

#박스그래프
data = [[2, 4, 6, 8, 10],[6 ,7, 8, 2, 4],[1, 3, 5, 7, 9],[7, 8, 2, 4, 2]]
df = pd.DataFrame(data)
plt.boxplot(df)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Box Plot')
plt.show()

#히트맵
data=[[2,3,4,1],[6,3,5,2],[6,3,5,4],[3,7,5,4],[2,8,1,5]] #5x4 matrix
df1 = pd.DataFrame(data)
plt.pcolor(df1.corr())
# plt.matshow(df1.corr())
plt.colorbar()

#파일 열어서 그래프그리기
os.getcwd()
os.chdir('D:\\TEACHING\\TEACHING KOREA\\SEOULTECH\\PYTHON\\2020 SPRING\\LABS')
data = pd.read_csv('graph_pd.txt')
data
data.columns = ['x', 'y']
plt.plot(data.x, data.y)
plt.show()

data1 = np.loadtxt('graph_np.txt', delimiter=',')
x = data1[0,:]
y = data1[1,:]
plt.plot(x,y)
plt.show()

#CHIPOTLE EXAMPLE
chipo = pd.read_excel('chipotle.xlsx')

# Scatter plot
chipo.columns
x = chipo.item_price
y = chipo.quantity
plt.scatter(x,y, color='k', s=100)
plt.xlabel('Item Price')
plt.ylabel('Quantity')
plt.title('Scatter Plot')
plt.show()

# Histogram
plt.hist(chipo.item_price, rwidth=.8)
plt.xlabel('Item Price')
plt.ylabel('Count')
plt.title('Histogram')
plt.show()

# Bar chart
freq = chipo.item_name.value_counts()
x = freq.index
y = freq.values
plt.bar(x,y, color='r')
plt.xlabel('Item Name')
plt.ylabel('Count')
plt.title('Bar Chart')
plt.xticks(rotation=90)
plt.show()

# Horizontal bar chart
plt.barh(x,y,color='r')
plt.ylabel('Item Name')
plt.xlabel('Count')
plt.title('Bar Chart')
plt.show()

# Heat map
plt.pcolor(chipo.corr())
plt.colorbar()

plt.style.available
plt.style.use('ggplot')

#subplot
names = ['a', 'b', 'c']
values = [1, 10, 100]
plt.figure(figsize=(9,3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot([1, 2, 3, 4, 5], [10, 5, 10, 5, 10], 'r-')
ax2 = fig.add_subplot(222)
ax2.plot([1, 2, 3, 4], [1,4,9,16], 'k-')
ax3 = fig.add_subplot(223)
ax3.plot([1, 2, 3, 4], [1,10,100,1000], 'b-')
ax4 = fig.add_subplot(224)
ax4.plot([1, 2, 3, 4], [0,0,1,1], 'g-')
plt.tight_layout()

# matplotlib auto
# matplotlib inline
raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'], 
        'age': [42, np.nan, 36, 24, 73], 
        'sex': ['m', np.nan, 'f', 'm', 'f'], 
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
df
df.isna().sum() #널값체크
df1 = df.dropna() #널값이 있는 행 제거
df1
df2 = df.dropna(how='all') #행전체가 널값인 경우 제거
df2
df['location']=np.nan
df.dropna(axis=1, how='all') #열전전체가 널값인 경우 제거
df.dropna(thresh=5) #널값이 아닌 것이 5개 이상이면 살림
df3 = df.fillna(0) #널값을 0으로 채움
df3
df.preTestScore.fillna(df.preTestScore.mean(), inplace=True)#널값을 점수의 평균값으로 채움

#CHIPOTLE 널값처리
chipo.isna().sum()
chipo1 = chipo.dropna()
chipo1.isna().sum()
plt.boxplot(chipo1.quantity)
plt.boxplot(chipo1.item_price)

#REPLACE TUTORIAL
df = pd.DataFrame({
    'name':['john','mary','paul'],
    'age':[30,25,40],
    'city':['new york','los angeles','london']
})
df.replace(25,40)
df.replace({
    25:26,
    'john':'johnny'})
df.replace('jo.+', 'FOO', regex=True)

df = pd.DataFrame({
    'name': ['john', 'mary', 'paul'],
    'num_children': [0, 4, 5],
    'num_pets':[0,1,2]})
df
df.replace({'num_pets':{0:1}})