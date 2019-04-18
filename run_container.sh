#!/bin/bash

# bind mount source code to allow testing on the fly
# without the need to rebuild container

docker container prune -f
docker run -d \
    -v /mnt/file_connector_write:/srv/fileconnector/srctest \
    -p 8083:8083 \
    -e FILE_CONNECTOR_SECRET=password \
    -e WRITE_METADATA=true \
    --name file-connector \
    file-connector:dev
