import requests
import bs4
from bs4 import BeautifulSoup

from .tools import ROOT

def collect_puppies():
    """
    Collect all entries from dogperday.com
    and write them to txt file.
    """
    pup_links = []

    # First page
    soup, links = scrap_links_page(
        url='https://dogperday.com/category/dog-of-the-day',
    )
    pup_links += links

    # Count pages:
    nav = soup.find(id="main").find("nav")
    text = next(nav.children).text
    pages = int(text.split()[-1])

    # Scrap all other pages:
    for i in range(2,pages+1):
        print(f'in page {i}')
        soup, links = scrap_links_page(
            url=f'https://dogperday.com/category/dog-of-the-day/page/{i}/',
        )
        pup_links += links

    with open(f'{ROOT}puppies/data/dogs_of_the_day.txt', 'w') as f:
        f.write('# See puupies/tools/pup_scrapper.py to create/update this list\n')
        for link in pup_links:
            f.write(f'{link}\n')


def scrap_links_page(url):
    """
    Scrap a https://dogperday.com webpage, extract links to dogperday entries.
    """
    headers = requests.utils.default_headers()
    headers['User-Agent'] = (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
    )
    response = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get all the links:
    dog_links = [
        link
        for link in soup.find(id="main").find_all("a")
        if link['href'].find("https://dogperday.com") != -1
        if 'wp' not in link['href']
        if link.text == ''
    ]
    links = [link['href'] for link in dog_links]

    return soup, links


