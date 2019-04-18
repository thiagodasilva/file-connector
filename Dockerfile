FROM swiftstack/picoswiftstack:6.19.0.1

MAINTAINER Thiago da Silva <thiago@swiftstack.com>

ARG FILE_CONNECTOR_DIR=.
COPY install_scripts /install_scripts
COPY $FILE_CONNECTOR_DIR/setup.py /opt/src/setup.py
COPY $FILE_CONNECTOR_DIR/requirements.txt /opt/src/requirements.txt
COPY $FILE_CONNECTOR_DIR/file_connector/ /opt/src/file_connector

RUN /install_scripts/install_prereqs.sh && \
    /install_scripts/fileconnector_install.sh

# Replace openstack swift conf files with local file-connector ones
COPY configs/swift/* /etc/swift/
COPY configs/rootfs /

ENTRYPOINT ["/init"]
