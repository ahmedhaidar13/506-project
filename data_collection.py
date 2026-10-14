# Pulls our data from the openFDA API and saves it to data/raw/
#
#   python data_collection.py            pull everything
#   python data_collection.py --codes    just check our product codes
#   python data_collection.py --n 2000   smaller sample for testing
#
# Set OPENFDA_API_KEY first to get 120k requests a day instead of 1k.

import argparse
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
        if r.status_code == 429 or r.status_code >= 500:  # rate limited or server hiccup, wait and retry
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


# search string for all of our devices' event reports in a date range (YYYYMMDD)
def date_search(start, end):
    dates = f"date_received:[{start}+TO+{end}]"
    return make_search(config.PRODUCT_CODES.keys(), dates,
                       field="device.device_report_product_code")


# how many reports each day of a year has, in one request (keys look like "20200131")
def daily_totals(year):
    data = get(EVENTS, {"search": date_search(f"{year}0101", f"{year}1231"),
                        "count": "date_received", "limit": 1000})
    return {row["time"]: row["count"] for row in data["results"]}


# Get a random sample of k event reports from one day
# Most months have over 25,000 reports, which is past openFDA's skip limit,
# so we sample by day instead (a day is well under the limit).
PAGE = 25


def sample_day(day, total, k, rng):
    search = date_search(day, day)
    pages = list(range(0, min(total, config.MAX_SKIP), PAGE))
    picked = rng.sample(pages, min(len(pages), -(-k // PAGE)))

    reports = []
    for skip in sorted(picked):
        # sort so pages come back in the same order every request
        reports += get(EVENTS, {"search": search, "limit": PAGE, "skip": skip})["results"]
        time.sleep(0.25)
    return rng.sample(reports, min(k, len(reports)))


# sample n adverse event reports spread evenly across every day in our date range
def pull_events(n):
    rng = random.Random(config.SEED)

    print("counting reports per day")
    totals = {}
    for year in range(config.START_YEAR, config.END_YEAR + 1):
        totals.update(daily_totals(year))
        time.sleep(0.25)
    save([{"date": d, "reports": t} for d, t in sorted(totals.items())], "daily_totals")

    too_big = [d for d, t in totals.items() if t > config.MAX_SKIP]
    if too_big:
        print(f"  WARNING: {len(too_big)} days have over {config.MAX_SKIP} reports, "
              f"those days can't be fully sampled: {sorted(too_big)[:10]}")

    # split n evenly across days, giving the leftover reports to random days
    days = sorted(d for d, t in totals.items() if t > 0)
    base, extra = divmod(n, len(days))
    bonus = set(rng.sample(days, extra))

    print(f"sampling {n} adverse event reports from {len(days)} days")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / "events_sample.jsonl"
    seen = set()
    # write as we go so a crash halfway through doesn't lose everything
    with open(path, "w") as f:
        for i, day in enumerate(days):
            k = base + (day in bonus)
            if k == 0:
                continue
            for r in sample_day(day, totals[day], k, rng):
                key = r.get("mdr_report_key")
                if key not in seen:  # skip duplicate reports
                    seen.add(key)
                    f.write(json.dumps(r) + "\n")
            if day.endswith("01"):
                print(f"  {day[:4]}-{day[4:6]}: {len(seen)} reports so far")
    print(f"  saved {len(seen)} records to {path.name}")


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