import pandas as pd
import selenium
from selenium import webdriver
from selenium.common import WebDriverException, TimeoutException
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json


def scrape_website(website):
    print(f"Connecting to {website}...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = ChromeService()

    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(15)  # макс. 15 секунд на загрузку страницы

        try:
            driver.get(website)
            time.sleep(2)
            return driver.page_source
        except (WebDriverException, TimeoutException) as e:
            print(f"[!] Failed to load {website}: {type(e).__name__} – {e}")
            return None
        finally:
            driver.quit()

    except Exception as e:
        print(f"[!] Driver init error: {e}")
        return None


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def extract_text_nodes_in_order(html_content, url):
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body

    allowed_tags = ['h1', 'h2', 'h3', 'h4', 'p', 'a', 'span', 'div', 'li']
    nodes = []

    for tag in body.find_all(allowed_tags):
        if tag.find():
            continue

        text = tag.get_text(separator="\n")
        text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        if text and len(text) < 200:
            nodes.append({"url": url, "tag": tag.name, "text": text, "label": 0})

    return nodes

def remove_duplicates_by_text(nodes):
    seen_texts = set()
    unique_nodes = []
    for node in nodes:
        if node["text"] not in seen_texts:
            seen_texts.add(node["text"])
            unique_nodes.append(node)

    return unique_nodes

def main():
    urls = pd.read_csv('data/checked_urls.csv')['url'].tolist()

    with open("data/product_dataset_with_urls.jsonl", "w", encoding="utf-8") as f:
        for url in urls:
            print(f"\nProcessing URL: {url}")
            try:
                html = scrape_website(url)
                if not html:
                    continue

                body = extract_body_content(html)
                nodes = extract_text_nodes_in_order(body, url)
                nodes = remove_duplicates_by_text(nodes)

                for item in nodes:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"[!] Error while processing {url}: {type(e).__name__} – {e}")
                continue

if __name__ == "__main__":
    main()