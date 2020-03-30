#!/bin/bash

# To print full url (incl token) after some time. DEV ONLY
sleep 5 && jupyter notebook list &

echo 'Starting jupyter lab'
jupyter lab --ip=0.0.0.0 \
			--port=8899 \
			--browser=false \
			--NotebookApp.token=$JUPYTER_TOKEN \
		   	--allow-root
