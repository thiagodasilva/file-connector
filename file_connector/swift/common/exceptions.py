# Copyright (c) 2012-2013 Red Hat, Inc.
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

from swift.common.exceptions import SwiftException


class FileConnectorFileSystemOSError(OSError):
    pass


class FileConnectorFileSystemIOError(IOError):
    pass


class FileConnectorException(Exception):
    pass


class FailureToMountError(FileConnectorException):
    pass


class FileOrDirNotFoundError(FileConnectorException):
    pass


class NotDirectoryError(FileConnectorException):
    pass


class AlreadyExistsAsDir(FileConnectorException):
    pass


class AlreadyExistsAsFile(FileConnectorException):
    pass


class DiskFileContainerDoesNotExist(FileConnectorException):
    pass


class ThreadPoolDead(SwiftException):
    pass
