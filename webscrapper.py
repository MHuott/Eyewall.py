#create a list with all the files links associated with URL

import requests

url = 'https://tropic.ssec.wisc.edu/archive/mimtc/2020_31L/web/data/'
page = requests.get(url).text

from bs4 import BeautifulSoup
soup = BeautifulSoup(page, 'html.parser')
links = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('nc')]

#Download all the files

import urllib.request




for link in links:
    print(link)
    filename = link.split('/')[-1]
    print(filename)
    urllib.request.urlretrieve(link,filename)
