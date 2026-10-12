# Pulls our data from the openFDA API and saves it to data/raw/
#
#   python data_collection.py            pull everything
#   python data_collection.py --codes    just check our product codes
#   python data_collection.py --n 2000   smaller sample for testing
#
# Set OPENFDA_API_KEY first to get 120k requests a day instead of 1k.

import argparse
import calendar
import json
import random
import time

import requests

import config

EVENTS = "/device/event.json"
RECALLS = "/device/recall.json"
CLEARANCES = "/device/510k.json"
CLASSIFICATION = "/device/classification.json"

# build a search string for the openFDA API
def make_search(codes, extra=None, field="product_code"):
    search = field + ":(" + "+".join(sorted(codes)) + ")"
    if extra:
        search += "+AND+" + extra
    return search

# get a single page of results from the openFDA API, retrying if we get rate limited
def get(path, params):
    if config.API_KEY:
        params["api_key"] = config.API_KEY
    # build the url by hand since openFDA needs the + and : characters as is
    url = config.API_BASE + path + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    for attempt in range(3):
        r = requests.get(url, timeout=60)
        if r.status_code == 404:  # openFDA returns 404 when nothing matches
            return {"meta": {"results": {"total": 0}}, "results": []}
        if r.status_code == 429:  # rate limited, wait and try again
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


# keep paging through results until we have everything (or hit openFDA's skip limit)
def get_all(path, search):
    results = []
    skip = 0
    while skip <= config.MAX_SKIP:
        data = get(path, {"search": search, "limit": config.PAGE_SIZE, "skip": skip})
        page = data["results"]
        results += page
        total = data["meta"]["results"]["total"]
        skip += config.PAGE_SIZE
        if not page or skip >= total:
            break
        time.sleep(0.25)
    if len(results) < total:
        print(f"  only got {len(results)} of {total}, try a smaller date range")
    return results


# Write records to data/raw/ as a jsonl file, one record per line
def save(records, name):
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / (name + ".jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  saved {len(records)} records to {path.name}")


def get_each_code(path):
    # search one product code at a time, these endpoints don't like grouped codes
    results = []
    for code in config.PRODUCT_CODES:
        results += get_all(path, make_search([code]))
        time.sleep(0.25)
    return results


# pull classifications, 510(k) clearances and recalls for our devices
def pull_device_info():
    print("getting device classifications")
    save(get_each_code(CLASSIFICATION), "classification")

    print("getting 510(k) clearances")
    save(get_each_code(CLEARANCES), "510k")

    print("getting recalls")
    recalls = get_each_code(RECALLS)
    save(recalls, "recall")
    return recalls


# Count how many recalls each product code has
def count_recalls(recalls):
    # one recall can cover more than one product, so count each recall number once
    seen = set()
    counts = {code: 0 for code in config.PRODUCT_CODES}
    for r in recalls:
        key = (r.get("product_code"), r.get("res_event_number"))
        if key not in seen and key[0] in counts:
            seen.add(key)
            counts[key[0]] += 1
    return counts


# search string for all of our devices' event reports received in a given month
def month_search(year, month):
    last_day = calendar.monthrange(year, month)[1]
    dates = f"date_received:[{year}{month:02d}01+TO+{year}{month:02d}{last_day}]"
    return make_search(config.PRODUCT_CODES.keys(), dates,
                       field="device.device_report_product_code")


# Get a random sample of n event reports from one month
PAGE = 500


def sample_month(year, month, n, rng):
    # grab random pages of 500 from the month instead of just the first n reports
    search = month_search(year, month)
    total = get(EVENTS, {"search": search, "limit": 1})["meta"]["results"]["total"]
    pages = list(range(0, min(total, config.MAX_SKIP), PAGE))
    picked = rng.sample(pages, min(len(pages), (n + PAGE - 1) // PAGE))

    reports = []
    for skip in picked:
        reports += get(EVENTS, {"search": search, "limit": PAGE, "skip": skip})["results"]
        time.sleep(0.25)
    return reports[:n], total


# sample n adverse event reports spread evenly across every month in our date range
def pull_events(n):
    print(f"sampling {n} adverse event reports")
    rng = random.Random(config.SEED)
    months = [(y, m) for y in range(config.START_YEAR, config.END_YEAR + 1)
              for m in range(1, 13)]
    per_month = n // len(months)

    reports = []
    for year, month in months:
        got, total = sample_month(year, month, per_month, rng)
        reports += got
        print(f"  {year}-{month:02d}: took {len(got)} of {total}")
    save(reports, "events_sample")


# Make sure our product codes exist and look for other codes we might be missing
def check_codes():
    found = get_each_code(CLASSIFICATION)
    by_code = {r["product_code"]: r for r in found}
    print("our product codes:")
    for code in config.PRODUCT_CODES:
        r = by_code.get(code)
        if r:
            print(f"  {code}: class {r.get('device_class')}, {r.get('device_name')}")
        else:
            print(f"  {code}: not found")

    print("other codes that might fit:")
    seen = set(config.PRODUCT_CODES)
    for term in config.SEARCH_TERMS:
        for r in get_all(CLASSIFICATION, 'device_name:"' + term.replace(" ", "+") + '"'):
            if r["product_code"] not in seen:
                seen.add(r["product_code"])
                print(f"  {r['product_code']}: class {r.get('device_class')}, {r.get('device_name')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", action="store_true")
    parser.add_argument("--n", type=int, default=config.SAMPLE_SIZE)
    args = parser.parse_args()

    if not config.API_KEY:
        print("no OPENFDA_API_KEY set, you only get 1000 requests a day")

    if args.codes:
        check_codes()
    else:
        recalls = pull_device_info()
        print("recalls per device:")
        for code, count in count_recalls(recalls).items():
            print(f"  {code} ({config.PRODUCT_CODES[code]}): {count}")
        pull_events(args.n)