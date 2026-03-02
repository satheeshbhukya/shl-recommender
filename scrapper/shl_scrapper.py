import requests
from bs4 import BeautifulSoup
import time
import json
import os
from tqdm import tqdm
import re

BASE_URL    = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
}

OUTPUT_PATH    = "output/shl_individual_tests.json"
ITEMS_PER_PAGE = 12

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def has_yes(td):
    return bool(td.select_one("span.catalogue__circle.-yes")) if td else False


def extract_rows_from_page(start):
    url  = f"{CATALOG_URL}?start={start}"
    soup = get_soup(url)
    target_table = None
    for table in soup.select("table"):
        header_th = table.find("th", string=lambda s: s and "Individual Test Solutions" in s)
        if header_th:
            target_table = table
            break
    if not target_table:
        print(" Could not find Individual Test Solutions table!")
        return []
    items = []
    for row in target_table.select("tr[data-entity-id]"):
        title_td = row.select_one("td.custom__table-heading__title")
        if not title_td:
            continue
        a_tag = title_td.find("a", href=True)
        if not a_tag:
            continue
        name = a_tag.get_text(strip=True)
        link = BASE_URL + a_tag["href"]
        general_tds      = row.select("td.custom__table-heading__general")
        remote_testing   = "Yes" if len(general_tds) > 0 and has_yes(general_tds[0]) else "No"
        adaptive_support = "Yes" if len(general_tds) > 1 and has_yes(general_tds[1]) else "No"
        keys_td   = row.select_one("td.product-catalogue__keys")
        test_type = []
        if keys_td:
            letters   = [s.get_text(strip=True) for s in keys_td.select("span.product-catalogue__key")]
            test_type = [TEST_TYPE_MAP.get(l, l) for l in letters]

        items.append({
            "name":             name,
            "url":              link,
            "remote_testing":   remote_testing,
            "adaptive_support": adaptive_support,
            "test_type":        test_type,
        })
    return items


def get_all_catalog_items(max_pages=100):
    all_items = []
    seen_urls = set()
    for page in range(max_pages):
        start = page * ITEMS_PER_PAGE
        items = extract_rows_from_page(start)
        if not items:
            print("Empty page — stopping.")
            break
        new = 0
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
                new += 1
        if new == 0:
            print("No new items — end of catalog.")
            break
        time.sleep(1.0)
    return all_items


def parse_test_page(url):
    soup = get_soup(url)
    description = ""
    duration    = 0
    for row in soup.select("div.product-catalogue-training-calendar__row"):
        h4 = row.find("h4")
        p  = row.find("p")
        if not (h4 and p):
            continue
        key   = h4.get_text(strip=True).lower()
        value = p.get_text(strip=True)
        if "description" in key:
            description = value
        elif "assessment length" in key:
            match    = re.search(r"=\s*(\d+)", value)
            duration = int(match.group(1)) if match else 0
    return {
        "description": description,
        "duration":    duration,
    }


def scrape_all():
    os.makedirs("output", exist_ok=True)
    catalog_items = get_all_catalog_items()

    results = []
    for item in tqdm(catalog_items, desc="Fetching detail pages"):
        try:
            details = parse_test_page(item["url"])
            results.append({
                "url":              item["url"],
                "name":             item["name"],
                "adaptive_support": item["adaptive_support"],
                "description":      details["description"],
                "duration":         details["duration"],
                "remote_testing":   item["remote_testing"],
                "test_type":        item["test_type"],
            })
            time.sleep(0.4)
        except Exception as e:
            print(f"Failed: {item['url']} | {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    scrape_all()