# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@author: Kim
"""

print('Hello, Python!')
print("Python is a really great language, ", "isn't it?")

# Twinkle 예제
print('Twinkle, twinkle, little star,')
print('\tHow I wonder what you are!')
print('\t\tUp above the world so high,')
print('\t\tLike a diamond in the sky.')
print('Twinkle, twinkle, little star,')
print('\tHow I wonder what you are!')

# 세금계산프로그램
rate = .2
std_ded = 10000
dep_ded = 3000

income = float(input('Enter the gross income: '))
num_dep = int(input('Enter the number of dependents: '))
tax = (income - std_ded - dep_ded * num_dep)*rate
print('The income tax is $' + str(tax))

# 100살이 되는 년도 계산
name = input("What is your name?")
age = input("How old are you?")
year = 100 - int(age) + 2018
print(name + " will be 100 years old in the year " + str(year))

# BMI 계산
weight = input("what is your weight?")
height = input("what is your height?")
bmi = int(weight) / int(height)**2
print("your BMI is " + "%6.4f"% bmi)
#print("your BMI is " + str(bmi))

