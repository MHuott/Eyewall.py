#create a list with all the files links associated with URL

from urllib.parse import urlparse
import requests
import os
import urllib.request
from bs4 import BeautifulSoup

urls = ['url']

for url in urls:

    try:
        page = requests.get(url).text

        soup = BeautifulSoup(page, 'html.parser')
        links = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('nc')]

        #Download all the files

        # Parse the URL
        parsed_url = urlparse(url)

        # Split the path to get the part we need
        path_parts = parsed_url.path.split('/')

        # Extract the part '2021_07E' (assuming it's always in the same position)
        folder_name = path_parts[3]  # 2021_07E is the fourth element (index 3)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)


            # Download each file
        for link in links:
            filename = os.path.join(folder_name, os.path.basename(link))  # Save with proper filename
            urllib.request.urlretrieve(link, filename)  # Download the file

    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
    except Exception as e:
        print(f"Error processing {url}: {e}")
