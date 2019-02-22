#!/bin/sh

cd /opt/file-connector && \
pip install -r requirements.txt && \
pip install -e . && \
fileconnector-gen-builders fileconnector
