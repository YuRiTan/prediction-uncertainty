SHELL := /bin/bash
cwd := $(shell pwd)
PYTHON_INTERPRETER := python3

venv: 
	$(PYTHON_INTERPRETER) -m venv venv

clean_venv:
	rm -rf venv

activate_venv: venv
	source venv/bin/activate

.ONESHELL:
requirements: activate_venv
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]

dataset: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

clean_build:
	find . -name "*.py[co]" -delete
	find . -name "__pycache__" -delete
	rm -rf dist build .eggs *.egg-info

clean_all: clean_build clean_venv
	find . -name ".ipynb_checkpoints" -exec rm -rf {} +

.ONESHELL:
jupyter: activate_venv
	jupyter lab