import requests
import pandas as pd


def check_url(url, timeout=10):
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 405 or response.status_code >= 400:
            response = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
        if response.status_code == 200:
            return True
    except requests.RequestException as e:
        return str(e.__class__.__name__)


def main():
    urls = pd.read_csv('data/url_list.csv')['url'].tolist()
    checked_urls = []
    for url in urls:
        if check_url(url):
            checked_urls.append(url)
    pd.DataFrame(checked_urls, columns=['url']).to_csv('data/checked_urls.csv')

if __name__ == '__main__':
    main()