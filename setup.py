# Copyright (c) 2013 Red Hat, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from file_connector.swift import _pkginfo


setup(
    name=_pkginfo.name,
    version=_pkginfo.full_version,
    description='Nas Connector',
    license='Apache License (2.0)',
    author='SwiftStack',
    packages=find_packages(exclude=['test', 'bin']),
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 5 - Production/Stable'
        'Environment :: OpenStack'
        'Intended Audience :: Information Technology'
        'Intended Audience :: System Administrators'
        'License :: OSI Approved :: Apache Software License'
        'Operating System :: POSIX :: Linux'
        'Programming Language :: Python'
        'Programming Language :: Python :: 2'
        'Programming Language :: Python :: 2.6'
        'Programming Language :: Python :: 2.7'
    ],
    install_requires=[],
    scripts=[
        'bin/fileconnector-gen-builders',
    ],
    entry_points={
        'paste.app_factory': [
            'proxy=file_connector.swift.proxy.server:app_factory',
            'object=file_connector.swift.obj.server:app_factory',
            'container=file_connector.swift.container.server:app_factory',
            'account=file_connector.swift.account.server:app_factory',
        ],
        'paste.filter_factory': [
            'file_auth=file_connector.swift.common.middleware.'
            'file_auth:filter_factory'
        ],
    },
)
