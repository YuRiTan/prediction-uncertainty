SHELL := /bin/bash
cwd := $(shell pwd)
PYTHON_INTERPRETER := python3
IMG_TAG := prediction_uncertainty
.DEFAULT_GOAL := build

local_venv:
	$(PYTHON_INTERPRETER) -m venv venv

.ONESHELL:
local_requirements: local_venv
	source venv/bin/activate
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]

.ONESHELL:
dataset: local_requirements
	source venv/bin/activate
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

.ONESHELL:
local_jupyter: 
	source venv/bin/activate
	jupyter lab

build:
	docker-compose build
	# docker build -t $(IMG_TAG) .

.ONESHELL:
jupyter:
	@if ! docker ps | grep $(IMG_TAG) &> /dev/null; then
		docker-compose up
	@else
		@echo 'Container $(IMG_TAG) already running.'
		@echo 'Run `make clean_docker` first befor starting a new one'
	@fi
	# docker run -v $(cwd):/workspace -ti $(IMG_TAG)

clean_local_venv:
	rm -rf venv

clean_local_build:
	find . -name "*.py[co]" -delete
	find . -name "__pycache__" -delete
	rm -rf dist build .eggs *.egg-info

clean_docker:
	docker-compose down

clean_all: clean_local_build clean_local_venv
	find . -name ".ipynb_checkpoints" -exec rm -rf {} +

flush_docker: clean_docker
	docker images -a | grep $(IMG_TAG) | awk '{print $$3}' | xargs docker rmi
	docker system prune -f
