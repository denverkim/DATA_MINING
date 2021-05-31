# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:25:27 2020

@author: ho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('D:\\TEACHING\\TEACHING KOREA\\SEOULTECH\\PYTHON\\2020 SPRING\\LABS')

# Read the world food bank tsv file and assign it to a dataframe called food and complete the following tasks. 월드푸드뱅크 데이터를 읽어서 food라는 데이터프레임에 저장한후 다음 작업을 수행하시요.
food = pd.read_csv('worldfoodfact.tsv', sep='\t')
a = food.describe()

# Check missing values, count them by columns, and count the total number of missing 널값 체크, 컬럼별로 갯수체크, 전체갯수
food.isna()
food.isna().sum()
food.isna().sum().sum()

# Drop all missing observations 관측치별로 전체가 널값인 값 제거
a = food.dropna(how='all')
# Drop columns where all cells in that column is NA 전체컬럼이 널값인 컬럼제거
a = food.dropna(how='all', axis=1)
# Fill NA with the means of each column 각 컬럼의 평균값으로 널값 채우기
a = food.fillna(food.mean())

#MERGE EXAMPLE
left = pd.DataFrame({ 'id':[1,2,3,4,5], 
                     'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
                     'subject_id':['sub1','sub2','sub4','sub6','sub5']}) 
right = pd.DataFrame( {'id':[1,2,3,4,5], 
                       'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
                       'subject_id':['sub2','sub4','sub3','sub6','sub5']}) 
pd.merge(left, right, on='subject_id', how='right') 
pd.merge(left, right, on='subject_id', how='left') 
pd.merge(left, right, on='subject_id', how='outer')
pd.merge(left, right, on='subject_id', how='inner')
pd.merge(left, right, on='subject_id')
pd.merge(left, right, on=['id', 'subject_id'])

#CONCAT EXAMPLE
df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number']) 
pd.concat([df1, df2])
pd.concat([df1, df2], axis=1)

#CONCAT TUTORIAL
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_a

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_b

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
df_n

df_new = pd.concat([df_a, df_b])
pd.concat([df_a, df_b], axis=1)
pd.merge(df_new, df_n, on='subject_id')
pd.merge(df_new, df_n, on='subject_id', how='outer')
pd.merge(df_new, df_n, on='subject_id', how='inner')
pd.merge(df_new, df_n, on='subject_id', how='left')
pd.merge(df_new, df_n, on='subject_id', how='right')
pd.merge(df_a, df_b, on='subject_id', how='inner', suffixes=('_left', '_right'))
pd.merge(df_a, df_b, right_index=True, left_index=True)

#pivot table example
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

table = pd.pivot_table(df, values='D', index=['A','B'],
               columns=['C'], aggfunc=np.sum, fill_value=0)
table.plot(kind='bar')
table = pd.pivot_table(df, values=['D','E'], index=['A','C'], 
                       aggfunc={'D': np.mean,
                                'E': [min, np.mean, max]})
table.plot(kind='bar')

#crosstab example
a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
              "bar", "bar", "foo", "foo", "foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one",
              "one", "two", "two", "two", "one"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
              "shiny", "dull", "shiny", "shiny", "shiny"],
             dtype=object)
table = pd.crosstab(a, [b, c], rownames=['a'],
            colnames=['b', 'c'])
table.plot(kind='bar', stacked=True, grid=True)

foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
foo
bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
foo
table = pd.crosstab(foo, bar, dropna=False)
table.plot(kind='bar', grid=True)

#APPLY EXAMPLE
df = pd.DataFrame([[4,9],]*3, columns=['A', 'B'])
df.apply(np.sqrt)
df.apply(np.sum, axis=0)
df.apply(np.sum, axis=1)

#MAP EXAMPLE
x = pd.Series([1,2,3], index=['one', 'two', 'three'])
x
y = pd.Series(['foo', 'bar', 'baz'], index=[1,2,3])
y
z = {1:'A', 2:'B', 3:'C'}
z
x.map(y)
x.map(z)

def myfunc(n):
    return len(n)
list(map(myfunc, ('apple', 'banana', 'cherry')))
x = map(lambda n: len(n), ('apple', 'banana', 'cherry'))
list(x)
len(('apple', 'banana', 'cherry'))

#DATA MANIPULATION TUTORIAL
data = pd.read_csv('train.csv', index_col='Loan_ID')
data.loc[(data.Gender=='Female') & (data.Education == 'Not Graduate') 
     & (data.Loan_Status == 'Y'),['Gender', 'Education', 'Loan_Status']]

def num_missing(x):
    return x.isna().sum()
data.apply(num_missing, axis=0)
data.apply(num_missing, axis=1)

data.Gender.fillna(data.Gender.value_counts().index[0], inplace=True)
data.Married.fillna(data.Married.value_counts().index[0], inplace=True)
data.Self_Employed.fillna(data.Self_Employed.value_counts().index[0], inplace=True)

pd.pivot_table(data, values=['LoanAmount'], 
               index=['Gender', 'Married', 'Self_Employed'],
               aggfunc=np.mean)
pd.crosstab(data.Credit_History, data.Loan_Status, margins=True)

prop_rates = pd.DataFrame({
    'rates':[1000, 5000, 12000],
    'Property_Area':['Rural', 'Semiurban', 'Urban']})
prop_rates
data_merged = pd.merge(data, prop_rates, on='Property_Area', how='left')
pd.pivot_table(data_merged, values='Credit_History',
               index=['Property_Area', 'rates'], 
               aggfunc=len)

data.sort_values(['ApplicantIncome', 'CoapplicantIncome'], ascending=False)[['ApplicantIncome', 'CoapplicantIncome']].head(10)

data.boxplot(column='ApplicantIncome', by='Loan_Status')
data.hist(column='ApplicantIncome', by='Loan_Status',
          bins=30)