# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:34:59 2020

@author: kim
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://dataquestio.github.io/web-scraping-pages/simple.html"
page = requests.get(url)
page
page.status_code
page.content
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.p
soup.find('p')
soup.p.text
soup.p.string
soup.p.get_text()

url = "http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html"
page = requests.get(url)
page
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.find_all('p')
soup.findAll('p')[0].text.strip()
soup.find_all(class_="outer-text")
soup.find_all(id="first")
soup.find_all('p', class_="outer-text")
soup.find_all('p', {"class":"outer-text"})
soup.select('div p')

#예제3
url = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())
soup.find_all('a')[1]['href']
soup.find_all('a')[1].get('href')

all_links = soup.find_all('a')
for link in all_links:
    print(link.get('href'))
    
all_tables = soup.find_all('table')
right_table = soup.find_all('table', class_='wikitable sortable plainrowheaders')
right_table
A = []
B = []
C = []
D = []
E = []
F = []
G = []
for row in right_table[0].find_all('tr'):
    cells = row.find_all('td')
    states = row.find_all('th')
    if len(cells) == 6:
        A.append(cells[0].text.strip())
        B.append(states[0].text.strip())
        C.append(cells[1].text.strip())
        D.append(cells[2].text.strip())
        E.append(cells[3].text.strip())
        F.append(cells[4].text.strip())
        G.append(cells[5].text.strip())
A
B
C
D
E
F
G

df = pd.DataFrame({'Number': A,
                   'State/UT':B,
                   'Admin_Capital':C,
                   'Legistlative_Capital':D,
                   'Judiciary_Capital':E,
                   'Year_Capital':F,
                   'Former_Capital':G})
df
df.to_excel('D:/TEACHING/TEACHING KOREA/SEOULTECH/CT/LAB/capital_table.xlsx')