from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sys import getsizeof


def get_site_html(url):
    try:
        html = requests.get(url)
    except HTTPError as e:
        return None
    try:
        bs = BeautifulSoup(html.content, 'html.parser')
        title = bs
    except AttributeError as e:
        return None
    return title


def get_data():
    site_html = get_site_html('https://covid19.ncdc.gov.ng/')

    the_table = site_html.find('table', {'id': 'custom1'})
    body_table = the_table.find_all('tr')
    table_head = body_table[0]
    # print(table_head)
    table_row = body_table[1:]
    # print(table_row)
    headings = [ht.get_text() for ht in table_head.find_all('th')]
    # print(headings)
    gey = []
    all_rows = [table_row[i].find_all('td') for i in range(len(table_row))]
    for j in all_rows:
        one_row = []
        for h in j:
            one_row.append(h.get_text().replace('\n', ''))
        gey.append(one_row)

    df = pd.DataFrame(data=gey, columns=headings)
    return df
