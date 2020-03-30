FROM python:3.7

# Create required folders:
WORKDIR /pip
WORKDIR /root/.jupyter
WORKDIR /workspace

# Install Jupyterlab and python 3 dependencies:
COPY requirements.txt /pip/requirements.txt
# setup.py to make -e install link to correct path, before mounting the volume to that path
COPY setup.py /workspace/setup.py 
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Install Python / Jupyterlab dependencies:
RUN apt-get update && \
	pip3 install -U pip setuptools wheel && \
	pip3 install -r /pip/requirements.txt && \
	pip3 install -e /workspace/. && \
	apt-get clean && \
    apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* /tmp/* && \
	rm /workspace/setup.py

