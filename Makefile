# Run "make help" to see every command.
PYTHON = .venv/bin/python
INSTALLED = .venv/.installed
RAW_EVENTS = data/raw/events_sample.jsonl
CLEAN_EVENTS = data/processed/events_clean.csv

.PHONY: help install data process charts model stage2 all test clean check-raw check-processed

help:
	@echo "make install  - create .venv and install dependencies"
	@echo "make data     - download data from openFDA (needs OPENFDA_API_KEY in .env, slow)"
	@echo "make process  - clean raw data into $(CLEAN_EVENTS)"
	@echo "make charts   - class balance and findings charts"
	@echo "make model    - Stage 1: decision tree, model comparison, K-Means (all devices + pumps only)"
	@echo "make stage2   - Stage 2: recall backtest and timeline chart"
	@echo "make all      - process + charts + model + stage2 (reproduces every result)"
	@echo "make test     - run the tests"
	@echo "make clean    - delete .venv and processed data (raw downloads are kept)"

# only reinstalls when requirements.txt changes
$(INSTALLED): requirements.txt
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $(INSTALLED)

install: $(INSTALLED)

check-raw:
	@test -f $(RAW_EVENTS) || (echo "Missing $(RAW_EVENTS). Run \"make data\" first (see README)."; exit 1)

check-processed:
	@test -f $(CLEAN_EVENTS) || (echo "Missing $(CLEAN_EVENTS). Run \"make process\" first."; exit 1)

data: install
	$(PYTHON) data_collection.py

process: install check-raw
	$(PYTHON) clean_data.py

charts: install check-processed
	$(PYTHON) class_balance.py
	$(PYTHON) findings.py

model: install check-processed
	$(PYTHON) model.py
	$(PYTHON) model.py --pumps
	$(PYTHON) compare_models.py
	$(PYTHON) compare_models.py --pumps
	$(PYTHON) kmeans.py
	$(PYTHON) kmeans.py --pumps

stage2: install check-processed
	$(PYTHON) stage2_backtest.py
	$(PYTHON) stage2_plot.py

all: process charts model stage2

test: install
	$(PYTHON) -m pytest tests/ -v --disable-warnings

clean:
	rm -rf .venv data/processed