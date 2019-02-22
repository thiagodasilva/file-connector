#FROM swiftstack/picoswiftstack:6.17.0.0
FROM pico-swiftstack:latest

MAINTAINER Thiago da Silva <thiago@swiftstack.com>

ARG FILE_CONNECTOR_DIR=.
COPY install_scripts /install_scripts
COPY $FILE_CONNECTOR_DIR/setup.py /opt/src/setup.py
COPY $FILE_CONNECTOR_DIR/requirements.txt /opt/src/requirements.txt
COPY $FILE_CONNECTOR_DIR/file_connector/ /opt/src/file_connector

RUN apk --no-cache add bash
RUN /install_scripts/fileconnector_install_pico.sh

# Replace openstack swift conf files with local file-connector ones
COPY configs/swift/* /etc/swift/
COPY configs/rootfs /

ENTRYPOINT ["/init"]
