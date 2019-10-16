SHELL := /bin/bash
cwd := $(shell pwd)
PYTHON_INTERPRETER := python3

venv:
	$(PYTHON_INTERPRETER) -m venv venv

.ONESHELL:
requirements: venv
	source venv/bin/activate
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]

.ONESHELL:
dataset: requirements
	source venv/bin/activate
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

.ONESHELL:
jupyter: 
	source venv/bin/activate
	jupyter lab

clean_venv:
	rm -rf venv

clean_build:
	find . -name "*.py[co]" -delete
	find . -name "__pycache__" -delete
	rm -rf dist build .eggs *.egg-info

clean_all: clean_build clean_venv
	find . -name ".ipynb_checkpoints" -exec rm -rf {} +