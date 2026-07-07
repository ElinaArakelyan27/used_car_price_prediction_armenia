import requests
import time
import random
import os
import pandas as pd
from bs4 import BeautifulSoup
import re

BASE = "https://auto.am/offer/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "hy,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    # warm up by visiting homepage first
    try:
        s.get("https://auto.am/", timeout=10)
        time.sleep(1)
    except:
        pass
    return s

session = make_session()

# =========================
# PRICE EXTRACTION
# =========================
def extract_price(soup):
    page_text = soup.get_text(" ", strip=True)

    # Check contractual in first 700 chars — catches "Պայմ." short form
    top_text = page_text[:700]
    if "Պայմ." in top_text or "Պայմ" in top_text or "պայմ" in top_text:
        return 0, "contractual"

    # Search top 700 chars first (price is near the top on auto.am)
    price_match = re.search(r'(\d[\d ]*?)\s*\$', top_text)
    if price_match:
        val = int(price_match.group(1).replace(" ", ""))
        if 1000 <= val <= 500_000:
            return val, "fixed"

    # Fallback: search whole page
    price_match = re.search(r'(\d[\d ]*?)\s*\$', page_text)
    if price_match:
        val = int(price_match.group(1).replace(" ", ""))
        if 1000 <= val <= 500_000:
            return val, "fixed"

    # AMD fallback
    amd_match = re.search(r'(\d[\d ]*?)\s*դր', page_text[:700])
    if amd_match:
        val = int(amd_match.group(1).replace(" ", "")) // 400
        if 1000 <= val <= 500_000:
            return val, "amd"

    return 0, "unknown"

# =========================
# MILEAGE
# =========================
def extract_mileage(text):
    km = re.search(r'Վազքը\s+([\d][\d ]*)\s*կմ', text)
    if km:
        return int(km.group(1).replace(" ", ""))
    miles = re.search(r'Վազքը\s+([\d][\d ]*)\s*մղոն?', text, re.I)
    if miles:
        return round(int(miles.group(1).replace(" ", "")) * 1.60934)
    return None

# =========================
# EXTRA FIELDS
# =========================
def extract_extra_fields(text):
    fields = {}
    m = re.search(r'([\d]+\.?[\d]*)\s*լ\b', text)
    if m:
        fields["engine_l"] = float(m.group(1))
    for fuel in ["Բենզին", "Դիզել", "Հիբրիդ", "Էլեկտրո", "Գազ"]:
        if fuel in text:
            fields["fuel"] = fuel
            break
    for tr in ["Ավտոմատ", "Մեխանիկական", "Վարիատոր", "Ռոբոտ"]:
        if tr in text:
            fields["transmission"] = tr
            break
    for dr in ["Առջևի", "Հետևի", "Լրիվ"]:
        if dr in text:
            fields["drive"] = dr
            break
    m = re.search(r'Գույնը\s+([^\n,\.]+)', text)
    if m:
        fields["color"] = m.group(1).strip()
    for body in ["Սեդան", "Հեչբեք", "Կուպե", "Կաբրիոլետ", "Ունիվերսալ",
                 "Ջիփ", "Մինիվեն", "Ֆուրգոն", "Պիկափ"]:
        if body in text:
            fields["body_type"] = body
            break
    return fields

# =========================
# PARSE SINGLE CAR
# =========================
def parse_car(car_id):
    global session
    url = BASE.format(car_id)
    try:
        r = session.get(url, timeout=12)

        # blocked or dead — refresh session and retry once
        if r.status_code in (403, 429, 503) or len(r.text) < 500:
            print(f"  ⚠ status={r.status_code} len={len(r.text)} — refreshing session")
            session = make_session()
            time.sleep(5)
            r = session.get(url, timeout=12)

        if r.status_code != 200:
            return None

        text = r.text

        # definitive 404
        if "Page Not Found" in text or len(text) < 500:
            return None

        soup = BeautifulSoup(text, "html.parser")
        full_text = soup.get_text(" ", strip=True)

        # page must have car content — check several signals
        has_content = any(x in full_text for x in ["Վազքը", "դր.", "դր ", "$", "կմ", "մղոն"])
        if not has_content:
            return None

        title_tag = soup.find("title")
        if not title_tag:
            return None
        title_text = title_tag.text.replace("Auto.am - ", "").replace("auto.am - ", "").strip()
        parts = title_text.split()
        if len(parts) < 3:
            return None

        year  = parts[0]
        brand = parts[1]
        model = " ".join(parts[2:])

        if not re.match(r'^(19|20)\d{2}$', year):
            return None

        price, price_type = extract_price(soup)
        mileage = extract_mileage(full_text)
        extra   = extract_extra_fields(full_text)

        return {
            "url":        url,
            "car_id":     car_id,
            "year":       int(year),
            "brand":      brand,
            "model":      model,
            "price_usd":  price,
            "price_type": price_type,
            "mileage_km": mileage,
            **extra
        }

    except Exception as e:
        return None

# =========================
# MAIN CRAWLER
# =========================
def main():
    start_id = 3217000
    end_id   = 3200000

    data  = []
    found = 0
    miss  = 0
    consecutive_miss = 0

    print("STARTING ID CRAWL...\n")

    for car_id in range(start_id, end_id, -1):
        car = parse_car(car_id)

        if car:
            data.append(car)
            found += 1
            consecutive_miss = 0
            price_str = f"${car['price_usd']}" if car['price_usd'] else car['price_type']
            print(f"[FOUND] {car_id} | {car['year']} {car['brand']} {car['model']} | {price_str} | total: {found}")
        else:
            miss += 1
            consecutive_miss += 1
            if miss % 20 == 0:
                print(f"[MISS x{miss}] last: {car_id}")

        # if 500 consecutive misses, we're probably blocked — pause
        if consecutive_miss >= 500:
            print(f"⚠ {consecutive_miss} consecutive misses — pausing 30s and refreshing session")
            session = make_session()
            time.sleep(30)
            consecutive_miss = 0

        time.sleep(random.uniform(1.0, 2.5))

        if found > 0 and found % 50 == 0:
            pd.DataFrame(data).to_csv("data/raw/car_listings_raw.csv", index=False)
            print("💾 checkpoint saved")

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="url")
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/car_listings_raw.csv", index=False)
    print(f"\nDONE — found: {found}, missed: {miss}")
    print(df.head())

if __name__ == "__main__":
    main()