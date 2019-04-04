#!/bin/bash

# run haproxy as load balancer
# make sure to bind mount your own haproxy config file
# see haproxy.cfg_sample file for a sample
docker run -d -p 8080:8080 --name my-running-haproxy -v /home/thiago/haproxy_config:/usr/local/etc/haproxy:ro haproxy

# run memcache
docker run -p 11211:11211 --name my-memcache -d memcached memcached -m 64

# run file connector containers
# change mountpoint to your NFS share
# change MEMCACHE_SERVERS to poing to your memcache container
docker run -d \
    -v /home/thiago/file_connector_write:/srv/fileconnector/srctest \
    -p 8083:8083 \
    -e FILE_CONNECTOR_SECRET=password \
    -e MEMCACHE_SERVERS=192.168.0.18:11211 \
    --name file-connector \
    thiagodasilva/file-connector:dev

docker run -d \
    -v /home/thiago/file_connector_write:/srv/fileconnector/srctest \
    -p 8084:8083 \
    -e FILE_CONNECTOR_SECRET=password \
    -e MEMCACHE_SERVERS=192.168.0.18:11211 \
    --name file-connector2 \
    thiagodasilva/file-connector:dev

docker run -d \
    -v /home/thiago/file_connector_write:/srv/fileconnector/srctest \
    -p 8085:8083 \
    -e FILE_CONNECTOR_SECRET=password \
    -e MEMCACHE_SERVERS=192.168.0.18:11211 \
    --name file-connector2 \
    thiagodasilva/file-connector:dev
