#!/bin/sh

cd /
rm -rf /build

# apk del linux-headers
# apk del gnupg
apk del git
apk del buil-base
# apk del curl
# apk del rsync
# apk del memcached
# apk del sqlite-dev
# apk del xfsprogs
apk del autoconf
apk del automake
apk del libtool
apk del zlib-dev
apk del libffi-dev
apk del libxslt-dev
# apk del libxslt
apk del libxml2-dev
# apk del libxml2
# apk del python
apk del python-dev
# apk del py-pip
# apk del py-nose
rm -rf /var/cache/apk/*
