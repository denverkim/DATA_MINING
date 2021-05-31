# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:21:16 2020

@author: kim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

# Split the text into individual words and create a frequency table and plot. 다음 글을 단어로 쪼개서 빈도테이블과 그래프 그리기
text = "Now, I understand that because it's an election season expectations for what we will achieve this year are low But, Mister Speaker, I appreciate the constructive approach that you and other leaders took at the end of last year to pass a budget and make tax cuts permanent for working\
families. So I hope we can work together this year on some bipartisan priorities like criminal justice reform and helping people who are battling prescription drug abuse and heroin abuse. So, who knows, we might surprise the cynics again"
words = word_tokenize(text)
fdist = FreqDist(words)
fdist.plot()
plt.show()

# Split the text into sentences and tokenize the sentences and count the number of words. Draw the bar plot. 먼저 문장으로 쪼개고 다시 문장을 단어로 쪼개서 문장별 단어의 갯수를 카운트하고 막대그래프 그리기
sent = sent_tokenize(text)
word_count = []
for s in sent:
    word_count.append(len(word_tokenize(s)))
word_count

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

plt.bar(np.arange(1,4), word_count)
plt.title('문장당 단어수')
plt.show()

import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from wordcloud import WordCloud, STOPWORDS
# 링크에 있는 텍스트를 이용해서 불용어처리, 어간추출, 문장부호를 제거한 후 워드클라우드를 그리시오.
url = 'http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/sotu_text/236.txt'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
text = soup.get_text()
words = word_tokenize(text)
ps = PorterStemmer()
filter_words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
filter_words = [w for w in filter_words if w not in string.punctuation]
filter_words = [w for w in filter_words if w not in ["us", "S", "n't"]]
text = ' '.join(filter_words)
wordcloud = WordCloud(background_color='white').generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()