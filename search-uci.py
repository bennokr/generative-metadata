import requests, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlencode

BASE = "https://archive.ics.uci.edu/datasets"

def get_mixed_names(area="Health and Medicine", skip=0, take=50):
    params = {
        "Area": area,
        "FeatureTypes": "Mixed",
        "Python": "true",
        "skip": skip,
        "take": take,
        "orderBy": "NumHits",
        "sort": "desc",
        "search": ""
    }
    url = f"{BASE}?{urlencode(params, doseq=True)}".replace('%20', '+')
    print(f'Requesting {url}')
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    # dataset links live under h2 > a
    items = set()
    for h2 in soup.select("h2 > a"):
        name = h2.get_text(strip=True)
        items.add(name)
    return items

list_url = 'https://archive.ics.uci.edu/api/datasets/list'

sets = requests.get(list_url, params={'area':"Health and Medicine"})
sets = sets.json().get('data')

mixed_names = get_mixed_names()
mixed_data = [d for d in sets if d['name'] in mixed_names]


from ucimlrepo import fetch_ucirepo 

for dataset in mixed_data:
    # fetch dataset 
    d = fetch_ucirepo(id=dataset['id']) 
    
    # data (as pandas dataframes) 
    X = d.data.features 
    y = d.data.targets 
    
    # metadata 
    print(d.metadata) 
    
    # variable information 
    print(d.variables) 
    break
