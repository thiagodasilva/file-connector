#!/bin/sh

adduser -D -H syslog && \
adduser -D swift && \
mkdir -p /opt/swift && \
mkdir /opt/bin && \
mkdir /etc/swift && \
mkdir /var/spool/rsyslog && \
mkdir -p /var/run/swift && \
mkdir -p /srv/fileconnector && \
chown -R swift:swift /srv/fileconnector && \
chown -R swift:swift /var/run/swift && \
chown -R swift:swift /etc/swift
