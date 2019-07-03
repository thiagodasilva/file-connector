#!/bin/bash

set -e

# prepare fcdata dir
rm -rf /tmp/fcdata
mkdir -p /tmp/fcdata/migration
mkdir -p /tmp/fcdata/sync

# The following method is cross-platform (OS X and Linux)
MYDIR=$(python -c 'import os,sys;print(os.path.dirname(os.path.realpath(sys.argv[1])))' $0)
cd "$MYDIR"

docker build -t file-connector .

echo stop file-connector
docker container stop file-connector 2>/dev/null ||:
echo rm file-connector
docker container rm file-connector 2>/dev/null ||:

echo run file-connector
docker run -d \
    -v `pwd`:/opt/src \
    --name file-connector \
    -p "${HOST_FC_PORT:-8083}:8083" \
    -v /tmp/fcdata/migration:/srv/fileconnector/fcdata_migration \
    -v /tmp/fcdata/sync:/srv/fileconnector/fcdata_sync \
    -e FILE_CONNECTOR_SECRET=password \
    file-connector
