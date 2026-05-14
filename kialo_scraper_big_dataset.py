
# kialo_all_in_one_scraper.py

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "kialo_large_dataset.csv"
SCROLL_PAUSE = 2

# -----------------------------
# CREATE DRIVER
# -----------------------------
def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

driver = create_driver()

# -----------------------------
# SCROLL
# -----------------------------
def scroll(times=30):
    for _ in range(times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

# -----------------------------
# EXTRACT LINKS
# -----------------------------
def extract_links():
    links = set()

    elements = driver.find_elements(
        By.CSS_SELECTOR,
        "a[href*='/should-'], a[href*='/is-']"
    )

    for el in elements:
        href = el.get_attribute("href")
        if href:
            links.add(href)

    return links

# -----------------------------
# COLLECT ALL URLS
# -----------------------------
def collect_all_urls():

    all_links = set()

    # 🔹 Explore
    sections = ["new", "popular", "latest", "hot"]

    for section in sections:
        print(f"\n➡️ Explore: {section}")

        driver.get(f"https://www.kialo.com/explore/{section}")
        time.sleep(5)

        scroll(40)
        all_links.update(extract_links())

        print(f"Collected: {len(all_links)}")

    # 🔹 Tags
    print("\n➡️ Tags")

    driver.get("https://www.kialo.com/tags")
    time.sleep(5)

    tag_links = set()

    elements = driver.find_elements(By.TAG_NAME, "a")
    for el in elements:
        href = el.get_attribute("href")
        if href and "/tags/" in href:
            tag_links.add(href)

    for tag in tag_links:
        print(f"\n➡️ {tag}")

        driver.get(tag)
        time.sleep(5)

        scroll(30)
        all_links.update(extract_links())

        print(f"Collected: {len(all_links)}")

    # 🔹 Search
    queries = ["education", "AI", "war", "health", "science", "law"]

    for q in queries:
        print(f"\n➡️ Search: {q}")

        driver.get(f"https://www.kialo.com/search?q={q}")
        time.sleep(5)

        scroll(30)
        all_links.update(extract_links())

        print(f"Collected: {len(all_links)}")

    return list(all_links)

# -----------------------------
# SCRAPE DEBATE
# -----------------------------
def scrape_debate(url):

    driver.get(url)
    time.sleep(3)

    scroll(10)

    data = []

    try:
        topic = driver.find_element(By.TAG_NAME, "h1").text
    except:
        topic = "Unknown"

    arguments = driver.find_elements(By.CSS_SELECTOR, "[class*='argument']")

    for arg in arguments:
        try:
            text = arg.text.strip()

            if len(text) < 10:
                continue

            side = "unknown"

            if "pro" in text.lower():
                side = "pro"
            elif "con" in text.lower():
                side = "con"

            data.append({
                "topic": topic,
                "side": side,
                "content": text
            })

        except:
            continue

    return data

# -----------------------------
# MAIN
# -----------------------------
print("🚀 Collecting URLs...")
urls = collect_all_urls()

print(f"\n🔥 TOTAL URLS: {len(urls)}")

all_data = []
visited = set()

print("\n🚀 Scraping ALL data...")

for i, url in enumerate(urls):

    if url in visited:
        continue

    visited.add(url)

    try:
        print(f"{i+1}/{len(urls)} -> {url}")

        debate_data = scrape_debate(url)
        print("Arguments:", len(debate_data))

        all_data.extend(debate_data)

    except Exception as e:
        print("⚠️ Error:", e)

        # restart driver
        try:
            driver.quit()
        except:
            pass

        driver = create_driver()
        continue

    time.sleep(1)

    # restart every 100
    if i % 100 == 0 and i != 0:
        driver.quit()
        driver = create_driver()

driver.quit()

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
unique = []
seen = set()

for d in all_data:
    key = (d['topic'], d['content'])

    if key not in seen:
        seen.add(key)
        unique.append(d)

# -----------------------------
# SAVE DATA
# -----------------------------
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["topic", "side", "content"]
    )

    writer.writeheader()
    writer.writerows(unique)

print("\n=================================")
print("FINAL DATASET SIZE:", len(unique))
print("Saved to:", CSV_FILE)
print("=================================")