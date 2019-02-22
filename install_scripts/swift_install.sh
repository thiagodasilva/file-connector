#!/bin/sh

echo ${SWIFT_VERSION}
git clone --branch ${SWIFT_VERSION} --single-branch --depth 1 https://github.com/${SWIFT_REPO} /opt/swift
cd /opt/swift && \
pip install -r requirements.txt && \
pip install -e .

cp /opt/swift/doc/saio/bin/* /opt/bin
# cp doc/saio/bin/* /opt/bin
chmod +x /opt/bin/*
sed -i "s/bash/sh/g" /opt/bin/*
sed -i "s/sudo //g" /opt/bin/*
mkdir /root/tmp
echo "export PATH=${PATH}:/opt/bin" >> /opt/.shrc
echo "export PYTHON_EGG_CACHE=/root/tmp" >> /opt/.shrc
echo "export ENV=/opt/.shrc" >> /opt/.profile
chmod +x /opt/.shrc
chmod +x /opt/.profile
