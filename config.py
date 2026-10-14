import os
from pathlib import Path

# folders
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# load secrets like OPENFDA_API_KEY from .env (not committed)
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

# years we look at
START_YEAR = 2020
END_YEAR = 2025
TRAIN_YEARS = range(2020, 2024)  # 2020 to 2023
TEST_YEARS = range(2024, 2026)   # 2024 and 2025

# FDA product codes for insulin pumps and glucose monitors
# run "python data_collection.py --codes" to double check these
PRODUCT_CODES = {
    "LZG": "insulin pump",
    "QFG": "insulin pump (ACE)",
    "QJI": "automated glycemic controller",
    "QBJ": "integrated CGM",
    "MDS": "invasive glucose sensor (CGM)",
    "NBW": "OTC blood glucose meter",
}

#Words used to look for product codes we might have missed
SEARCH_TERMS = ["insulin pump", "infusion pump insulin", "glucose monitor",
                "continuous glucose", "glycemic controller"]

# openFDA settings
API_BASE = "https://api.fda.gov"
API_KEY = os.environ.get("OPENFDA_API_KEY", "")
PAGE_SIZE = 1000   # most records openFDA gives per request
MAX_SKIP = 25000   # openFDA won't let you skip past this

# how many adverse event reports to sample
SAMPLE_SIZE = 100000
SEED = 506

# labels for stage 1
POSITIVE = {"Injury", "Death"}
NEGATIVE = {"Malfunction"}