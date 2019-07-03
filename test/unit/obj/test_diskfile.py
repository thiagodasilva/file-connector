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

""" Tests for file_connector.swift.obj.diskfile """

import os
import stat
import errno
import mock
import re
import shutil
import tempfile
import unittest


import file_connector.swift.common.utils
import file_connector.swift.obj.diskfile

from eventlet import tpool, timeout, hubs
from mock import Mock, patch
from hashlib import md5
from copy import deepcopy
from contextlib import nested

from swift.common.exceptions import DiskFileNoSpace, DiskFileNotOpen, \
    DiskFileNotExist, DiskFileExpired
from swift.common.splice import splice
from swift.common.utils import get_md5_socket

from file_connector.swift.common.utils import ThreadPool
from file_connector.swift.common.exceptions import AlreadyExistsAsDir, \
    AlreadyExistsAsFile
from file_connector.swift.common.utils import normalize_timestamp, FC_ETAG
from file_connector.swift.obj.diskfile import DiskFileWriter, DiskFileManager
from file_connector.swift.common.utils import DEFAULT_UID, DEFAULT_GID, \
    X_OBJECT_TYPE, DIR_OBJECT

from test.unit.common.test_utils import _initxattr, _destroyxattr
from test.unit import FakeLogger

_metadata = {}


def _create_and_get_diskfile(df_mgr, td, dev, par, acc, con, obj, fsize=256):
    # FIXME: assumes account === volume
    the_path = os.path.join(td, dev, con)
    the_file = os.path.join(the_path, obj)
    base_obj = os.path.basename(the_file)
    base_dir = os.path.dirname(the_file)
    os.makedirs(base_dir)
    with open(the_file, "wb") as fd:
        fd.write("y" * fsize)
    gdf = _get_diskfile(df_mgr, dev, par, acc, con, obj)
    assert gdf._obj == base_obj
    assert gdf._fd is None
    return gdf


def _mapit(filename_or_fd):
    if isinstance(filename_or_fd, int):
        statmeth = os.fstat
    else:
        statmeth = os.lstat
    stats = statmeth(filename_or_fd)
    return stats.st_ino


def _mock_read_metadata(path, fd=None):
    global _metadata
    if fd:
        path = fd
    ino = _mapit(path)
    if ino in _metadata:
        md = _metadata[ino]
    else:
        md = {}
    return md


def _mock_write_metadata(path, metadata, fd=None):
    global _metadata
    if fd:
        path = fd
    ino = _mapit(path)
    _metadata[ino] = metadata


def _get_diskfile(df_mgr, d, p, a, c, o, **kwargs):
    df = df_mgr.get_diskfile(d, p, a, c, o, **kwargs)
    df._mp.write_metadata = _mock_write_metadata
    df._mp.read_metadata = _mock_read_metadata
    return df


def _mock_clear_metadata():
    global _metadata
    _metadata = {}


class MockException(Exception):
    pass


def _mock_rmobjdir(p):
    raise MockException("file_connector.swift.obj.diskfile.rmobjdir() called")


def _mock_do_fsync(fd):
    return


class MockRenamerCalled(Exception):
    pass


def _mock_renamer(a, b):
    raise MockRenamerCalled()


class TestDiskFileWriter(unittest.TestCase):
    """ Tests for file_connector.swift.obj.diskfile.DiskFileWriter """

    def setUp(self):
        self.lg = FakeLogger()
        self.td = tempfile.mkdtemp()
        self.conf = dict(devices=self.td, mb_per_sync=2,
                         keep_cache_size=(1024 * 1024), mount_check=False)
        self.mgr = DiskFileManager(self.conf, self.lg)

    def test_open_close(self):
        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "acc", "foo", "bar/obj")

        dw = DiskFileWriter('obj', 'foo/bar', '/dev/acc/foo/bar', 10, gdf)
        mock_open = Mock(return_value=100)
        mock_close = Mock()
        with nested(
                patch("file_connector.swift.obj.diskfile.do_open", mock_open),
                patch("file_connector.swift.obj.diskfile.do_close",
                      mock_close)):
            dw.open()
            self.assertEqual(100, dw._fd)
            self.assertEqual(1, mock_open.call_count)
            self.assertIn('/dev/acc/foo/bar/.obj.', dw._tmppath)

            dw.close()
            self.assertEqual(1, mock_close.call_count)
            self.assertEqual(None, dw._fd)

            # It should not call do_close since it should
            # have made fd equal to None
            mock_close.reset_mock()
            dw.close()
            self.assertEqual(None, dw._fd)
            self.assertEqual(0, mock_close.call_count)


class TestDiskFile(unittest.TestCase):
    """ Tests for file_connector.swift.obj.diskfile """

    def setUp(self):
        self._orig_tpool_exc = tpool.execute
        tpool.execute = lambda f, *args, **kwargs: f(*args, **kwargs)
        self.lg = FakeLogger()
        _initxattr()
        _mock_clear_metadata()
        #self._saved_df_wm = file_connector.swift.obj.diskfile.write_metadata
        #self._saved_df_rm = file_connector.swift.obj.diskfile.read_metadata
        #file_connector.swift.obj.diskfile.write_metadata = _mock_write_metadata
        #file_connector.swift.obj.diskfile.read_metadata = _mock_read_metadata
        #self._saved_ut_wm = file_connector.swift.common.utils.write_metadata
        #self._saved_ut_rm = file_connector.swift.common.utils.read_metadata
        #file_connector.swift.common.utils.write_metadata = _mock_write_metadata
        #file_connector.swift.common.utils.read_metadata = _mock_read_metadata
        self._saved_do_fsync = file_connector.swift.obj.diskfile.do_fsync
        file_connector.swift.obj.diskfile.do_fsync = _mock_do_fsync
        self.td = tempfile.mkdtemp()
        self.conf = dict(devices=self.td, mb_per_sync=2,
                         keep_cache_size=(1024 * 1024), mount_check=False)
        self.mgr = DiskFileManager(self.conf, self.lg)

    def tearDown(self):
        tpool.execute = self._orig_tpool_exc
        self.lg = None
        self.mgr = None
        _destroyxattr()
        #file_connector.swift.obj.diskfile.write_metadata = self._saved_df_wm
        #file_connector.swift.obj.diskfile.read_metadata = self._saved_df_rm
        #file_connector.swift.common.utils.write_metadata = self._saved_ut_wm
        #file_connector.swift.common.utils.read_metadata = self._saved_ut_rm
        file_connector.swift.obj.diskfile.do_fsync = self._saved_do_fsync
        shutil.rmtree(self.td)

    def test_constructor_no_slash(self):
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._mgr is self.mgr
        assert gdf._device_path == os.path.join(self.td, "vol0")
        assert isinstance(gdf._threadpool, ThreadPool)
        assert gdf._uid == DEFAULT_UID
        assert gdf._gid == DEFAULT_GID
        assert gdf._obj == "z"
        assert gdf._obj_path == ""
        self.assertEqual(
            gdf._put_datadir,
            os.path.join(self.td, "vol0", "bar"), gdf._put_datadir)
        assert gdf._data_file == os.path.join(self.td, "vol0", "bar", "z")
        assert gdf._fd is None

    def test_constructor_leadtrail_slash(self):
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "/b/a/z/")
        assert gdf._obj == "z"
        assert gdf._obj_path == "b/a"
        self.assertEqual(
            gdf._put_datadir,
            os.path.join(self.td, "vol0", "bar", "b", "a"), gdf._put_datadir)

    def test_open_no_metadata(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        stats = os.stat(the_file)
        ts = normalize_timestamp(stats.st_ctime)
        # ETag is calculated only after first read
        exp_md = {
            'Content-Length': 4,
            'ETag': FC_ETAG,
            'X-Timestamp': ts,
            'Content-Type': 'application/octet-stream'}
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._fd is None
        assert gdf._disk_file_open is False
        assert gdf._metadata is None
        with gdf.open():
            assert gdf._data_file == the_file
            assert gdf._fd is not None
            assert gdf._metadata == exp_md
            assert gdf._disk_file_open is True
        assert gdf._disk_file_open is False
        self.assertRaises(DiskFileNotOpen, gdf.get_metadata)
        self.assertRaises(DiskFileNotOpen, gdf.reader)
        self.assertRaises(DiskFileNotOpen, gdf.__enter__)

    def test_read_metadata_optimize_open_close(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        init_md = {
            'X-Type': 'Object',
            'X-Object-Type': 'file',
            'Content-Length': 4,
            'ETag': md5("1234").hexdigest(),
            'X-Timestamp': normalize_timestamp(os.stat(the_file).st_ctime),
            'Content-Type': 'application/octet-stream'}
        _metadata[_mapit(the_file)] = init_md
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._fd is None
        assert gdf._disk_file_open is False
        assert gdf._metadata is None

        # Case 1
        # Ensure that reading metadata for non-GET requests
        # does not incur opening and closing the file when
        # metadata is NOT stale.
        mock_open = Mock()
        mock_close = Mock()
        with mock.patch("file_connector.swift.obj.diskfile.do_open",
                        mock_open):
            with mock.patch("file_connector.swift.obj.diskfile.do_close",
                            mock_close):
                md = gdf.read_metadata()
                self.assertEqual(md, init_md)
        self.assertFalse(mock_open.called)
        self.assertFalse(mock_close.called)

        # Case 2
        # Ensure that reading metadata for non-GET requests
        # still opens and reads the file when metadata is stale
        with open(the_file, "a") as fd:
            # Append to the existing file to make the stored metadata
            # invalid/stale.
            fd.write("5678")
        md = gdf.read_metadata()
        # Check that the stale metadata is recalculated to account for
        # change in file content
        self.assertNotEqual(md, init_md)
        self.assertEqual(md['Content-Length'], 8)
        # ETag is calculated only after first read
        self.assertEqual(md['ETag'], FC_ETAG)

    def test_open_and_close(self):
        mock_close = Mock()

        with mock.patch("file_connector.swift.obj.diskfile.do_close",
                        mock_close):
            gdf = _create_and_get_diskfile(
                self.mgr, self.td, "vol0", "p57", "ufo47", "bar", "z")
            with gdf.open():
                assert gdf._fd is not None
            self.assertTrue(mock_close.called)

    def test_open_existing_metadata(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        ini_md = {
            'X-Type': 'Object',
            'X-Object-Type': 'file',
            'Content-Length': 4,
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'Content-Type': 'application/loctet-stream'}
        _metadata[_mapit(the_file)] = ini_md
        exp_md = ini_md.copy()
        del exp_md['X-Type']
        del exp_md['X-Object-Type']
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._fd is None
        assert gdf._metadata is None
        assert gdf._disk_file_open is False
        with gdf.open():
            assert gdf._data_file == the_file
            assert gdf._fd is not None
            self.assertEqual(
                gdf._metadata, exp_md, "%r != %r" % (gdf._metadata, exp_md))
            assert gdf._disk_file_open is True
        assert gdf._disk_file_open is False

    def test_open_invalid_existing_metadata(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        inv_md = {
            'Content-Length': 5,
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'Content-Type': 'application/loctet-stream'}
        _metadata[_mapit(the_file)] = inv_md
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._fd is None
        assert gdf._disk_file_open is False
        with gdf.open():
            assert gdf._data_file == the_file
            assert gdf._metadata != inv_md
            assert gdf._disk_file_open is True
        assert gdf._disk_file_open is False

    def test_open_isdir(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "d")
        os.makedirs(the_dir)
        ini_md = {
            'X-Type': 'Object',
            'X-Object-Type': 'dir',
            'Content-Length': 5,
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'Content-Type': 'application/loctet-stream'}
        _metadata[_mapit(the_dir)] = ini_md
        exp_md = ini_md.copy()
        del exp_md['X-Type']
        del exp_md['X-Object-Type']
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "d")
        assert gdf._obj == "d"
        assert gdf._disk_file_open is False
        with gdf.open():
            assert gdf._data_file == the_dir
            assert gdf._disk_file_open is True
        assert gdf._disk_file_open is False

    def test_reader(self):
        closed = [False]
        fd = [-1]

        def mock_close(*args, **kwargs):
            closed[0] = True
            os.close(fd[0])

        with mock.patch("file_connector.swift.obj.diskfile.do_close",
                        mock_close):
            gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0",
                                           "p57", "ufo47", "bar", "z")
            with gdf.open():
                assert gdf._fd is not None
                assert gdf._data_file == os.path.join(self.td, "vol0",
                                                      "bar", "z")
                reader = gdf.reader()
            assert reader._fd is not None
            fd[0] = reader._fd
            chunks = [ck for ck in reader]
            assert reader._fd is None
            assert closed[0]
            assert len(chunks) == 1, repr(chunks)

    def test_reader_disk_chunk_size(self):
        conf = dict(disk_chunk_size=64)
        conf.update(self.conf)
        self.mgr = DiskFileManager(conf, self.lg)
        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "ufo47", "bar", "z")
        with gdf.open():
            reader = gdf.reader()
        try:
            assert reader._disk_chunk_size == 64
            chunks = [ck for ck in reader]
        finally:
            reader.close()
        assert len(chunks) == 4, repr(chunks)
        for chunk in chunks:
            assert len(chunk) == 64, repr(chunks)

    def test_reader_iter_hook(self):
        called = [0]

        def mock_sleep(*args, **kwargs):
            called[0] += 1

        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "ufo47", "bar", "z")
        with gdf.open():
            reader = gdf.reader(iter_hook=mock_sleep)
        try:
            chunks = [ck for ck in reader]
        finally:
            reader.close()
        assert len(chunks) == 1, repr(chunks)
        assert called[0] == 1, called

    def test_reader_larger_file(self):
        closed = [False]
        fd = [-1]

        def mock_close(*args, **kwargs):
            closed[0] = True
            os.close(fd[0])

        with mock.patch("file_connector.swift.obj.diskfile.do_close",
                        mock_close):
            gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                           "ufo47", "bar", "z",
                                           fsize=1024*1024*2)
            with gdf.open():
                assert gdf._fd is not None
                self.assertEqual(gdf._data_file,
                                 os.path.join(self.td, "vol0", "bar", "z"))
                reader = gdf.reader()
            assert reader._fd is not None
            fd[0] = reader._fd
            chunks = [ck for ck in reader]
            assert reader._fd is None
            assert closed[0]

    def test_reader_dir_object(self):
        called = [False]

        def our_do_close(fd):
            called[0] = True
            os.close(fd)

        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(os.path.join(the_cont, "dir"))
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        with gdf.open():
            reader = gdf.reader()
        try:
            chunks = [ck for ck in reader]
            assert len(chunks) == 0, repr(chunks)
            with mock.patch("file_connector.swift.obj.diskfile.do_close",
                            our_do_close):
                reader.close()
            assert not called[0]
        finally:
            reader.close()

    def test_create_dir_object_no_md(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        the_dir = "dir"
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar",
                            os.path.join(the_dir, "z"))
        # Not created, dir object path is different, just checking
        assert gdf._obj == "z"
        wrt = gdf.writer()
        wrt._create_dir_object(the_dir)
        full_dir_path = os.path.join(the_cont, the_dir)
        assert os.path.isdir(full_dir_path)
        assert _mapit(full_dir_path) not in _metadata

    def test_create_dir_object_with_md(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        the_dir = "dir"
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar",
                            os.path.join(the_dir, "z"))
        # Not created, dir object path is different, just checking
        assert gdf._obj == "z"
        dir_md = {'Content-Type': 'application/directory',
                  X_OBJECT_TYPE: DIR_OBJECT}
        wrt = gdf.writer()
        wrt._create_dir_object(the_dir, dir_md)
        full_dir_path = os.path.join(the_cont, the_dir)
        assert os.path.isdir(full_dir_path)
        assert _mapit(full_dir_path) in _metadata

    def test_create_dir_object_exists(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "dir")
        os.makedirs(the_path)
        with open(the_dir, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir/z")
        # Not created, dir object path is different, just checking
        assert gdf._obj == "z"

        def _mock_do_chown(p, u, g):
            assert u == DEFAULT_UID
            assert g == DEFAULT_GID

        dc = file_connector.swift.obj.diskfile.do_chown
        file_connector.swift.obj.diskfile.do_chown = _mock_do_chown
        wrt = gdf.writer()
        self.assertRaises(
            AlreadyExistsAsFile, wrt._create_dir_object, the_dir)
        file_connector.swift.obj.diskfile.do_chown = dc
        self.assertFalse(os.path.isdir(the_dir))
        self.assertFalse(_mapit(the_dir) in _metadata)

    def test_create_dir_object_do_stat_failure(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "dir")
        os.makedirs(the_path)
        with open(the_dir, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir/z")
        # Not created, dir object path is different, just checking
        assert gdf._obj == "z"

        def _mock_do_chown(p, u, g):
            assert u == DEFAULT_UID
            assert g == DEFAULT_GID

        dc = file_connector.swift.obj.diskfile.do_chown
        file_connector.swift.obj.diskfile.do_chown = _mock_do_chown
        wrt = gdf.writer()
        self.assertRaises(
            AlreadyExistsAsFile, wrt._create_dir_object, the_dir)
        file_connector.swift.obj.diskfile.do_chown = dc
        self.assertFalse(os.path.isdir(the_dir))
        self.assertFalse(_mapit(the_dir) in _metadata)

    def test_write_metadata(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "z")
        os.makedirs(the_dir)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        md = {'Content-Type': 'application/octet-stream', 'a': 'b'}
        gdf.write_metadata(md.copy())
        self.assertEqual(None, gdf._metadata)
        fmd = _metadata[_mapit(the_dir)]
        md.update({'X-Object-Type': 'file', 'X-Type': 'Object'})
        self.assertTrue(fmd['a'], md['a'])
        self.assertTrue(fmd['Content-Type'], md['Content-Type'])

    def test_add_metadata_to_existing_file(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        ini_md = {
            'X-Type': 'Object',
            'X-Object-Type': 'file',
            'Content-Length': 4,
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'Content-Type': 'application/loctet-stream'}
        _metadata[_mapit(the_file)] = ini_md
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        md = {'Content-Type': 'application/octet-stream', 'a': 'b'}
        gdf.write_metadata(md.copy())
        self.assertTrue(_metadata[_mapit(the_file)]['a'], 'b')
        newmd = {'X-Object-Meta-test':'1234'}
        gdf.write_metadata(newmd.copy())
        on_disk_md = _metadata[_mapit(the_file)]
        self.assertTrue(on_disk_md['Content-Length'], 4)
        self.assertTrue(on_disk_md['X-Object-Meta-test'], '1234')
        self.assertTrue(on_disk_md['X-Type'], 'Object')
        self.assertTrue(on_disk_md['X-Object-Type'], 'file')
        self.assertTrue(on_disk_md['ETag'], 'etag')
        self.assertFalse('a' in on_disk_md)

    def test_add_md_to_existing_file_with_md_in_gdf(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        ini_md = {
            'X-Type': 'Object',
            'X-Object-Type': 'file',
            'Content-Length': 4,
            'name': 'z',
            'ETag': 'etag',
            'X-Timestamp': 'ts'}
        _metadata[_mapit(the_file)] = ini_md
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")

        # make sure gdf has the _metadata
        gdf.open()
        md = {'a': 'b'}
        gdf.write_metadata(md.copy())
        self.assertTrue(_metadata[_mapit(the_file)]['a'], 'b')
        newmd = {'X-Object-Meta-test':'1234'}
        gdf.write_metadata(newmd.copy())
        on_disk_md = _metadata[_mapit(the_file)]
        self.assertTrue(on_disk_md['Content-Length'], 4)
        self.assertTrue(on_disk_md['X-Object-Meta-test'], '1234')
        self.assertFalse('a' in on_disk_md)

    def test_add_metadata_to_existing_dir(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_cont, "dir")
        os.makedirs(the_dir)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        self.assertEquals(gdf._metadata, None)
        init_md = {
            'X-Type': 'Object',
            'Content-Length': 0,
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'X-Object-Meta-test':'test',
            'Content-Type': 'application/directory'}
        _metadata[_mapit(the_dir)] = init_md

        md = {'X-Object-Meta-test':'test'}
        gdf.write_metadata(md.copy())
        self.assertEqual(_metadata[_mapit(the_dir)]['X-Object-Meta-test'],
                'test')
        self.assertEqual(_metadata[_mapit(the_dir)]['Content-Type'].lower(),
                'application/directory')

        # set new metadata
        newmd = {'X-Object-Meta-test2':'1234'}
        gdf.write_metadata(newmd.copy())
        self.assertEqual(_metadata[_mapit(the_dir)]['Content-Type'].lower(),
                'application/directory')
        self.assertEqual(_metadata[_mapit(the_dir)]["X-Object-Meta-test2"],
                '1234')
        self.assertEqual(_metadata[_mapit(the_dir)]['X-Object-Type'],
                DIR_OBJECT)
        self.assertFalse('X-Object-Meta-test' in _metadata[_mapit(the_dir)])

    def test_write_metadata_w_meta_file(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        newmd = deepcopy(gdf.read_metadata())
        newmd['X-Object-Meta-test'] = '1234'
        gdf.write_metadata(newmd)
        assert _metadata[_mapit(the_file)] == newmd

    def test_write_metadata_w_meta_file_no_content_type(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        newmd = deepcopy(gdf.read_metadata())
        newmd['Content-Type'] = ''
        newmd['X-Object-Meta-test'] = '1234'
        gdf.write_metadata(newmd)
        assert _metadata[_mapit(the_file)] == newmd

    def test_write_metadata_w_meta_dir(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "dir")
        os.makedirs(the_dir)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        newmd = deepcopy(gdf.read_metadata())
        newmd['X-Object-Meta-test'] = '1234'
        gdf.write_metadata(newmd)
        assert _metadata[_mapit(the_dir)] == newmd

    def test_write_metadata_w_marker_dir(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "dir")
        os.makedirs(the_dir)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        newmd = deepcopy(gdf.read_metadata())
        newmd['X-Object-Meta-test'] = '1234'
        gdf.write_metadata(newmd)
        assert _metadata[_mapit(the_dir)] == newmd

    def test_put_w_marker_dir_create(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_cont, "dir")
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        assert gdf._metadata is None
        newmd = {
            'ETag': 'etag',
            'X-Timestamp': 'ts',
            'Content-Type': 'application/directory'}
        with gdf.create() as dw:
            dw.put(newmd)
        assert gdf._data_file == the_dir
        for key, val in newmd.items():
            assert _metadata[_mapit(the_dir)][key] == val
        assert _metadata[_mapit(the_dir)][X_OBJECT_TYPE] == DIR_OBJECT

    def test_put_is_dir(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "dir")
        os.makedirs(the_dir)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir")
        with gdf.open():
            origmd = gdf.get_metadata()
        origfmd = _metadata[_mapit(the_dir)]
        newmd = deepcopy(origmd)
        # FIXME: This is a hack to get to the code-path; it is not clear
        # how this can happen normally.
        newmd['Content-Type'] = ''
        newmd['X-Object-Meta-test'] = '1234'
        with gdf.create() as dw:
            try:
                # FIXME: We should probably be able to detect in .create()
                # when the target file name already exists as a directory to
                # avoid reading the data off the wire only to fail as a
                # directory.
                dw.write('12345\n')
                dw.put(newmd)
            except AlreadyExistsAsDir:
                pass
            else:
                self.fail("Expected to encounter"
                          " 'already-exists-as-dir' exception")
        with gdf.open():
            assert gdf.get_metadata() == origmd
        assert _metadata[_mapit(the_dir)] == origfmd, "was: %r, is: %r" % (
            origfmd, _metadata[_mapit(the_dir)])

    def test_put(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        self.assertEqual(gdf._obj, "z")
        self.assertEqual(gdf._obj_path, "")
        self.assertEqual(gdf._container_path,
                         os.path.join(self.td, "vol0", "bar"))
        self.assertEqual(gdf._put_datadir, the_cont)
        self.assertEqual(gdf._data_file,
                         os.path.join(self.td, "vol0", "bar", "z"))

        body = '1234\n'
        etag = md5()
        etag.update(body)
        etag = etag.hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        with gdf.create() as dw:
            self.assertIsNotNone(dw._tmppath)
            tmppath = dw._tmppath
            dw.write(body)
            dw.put(metadata)

        self.assertTrue(os.path.exists(gdf._data_file))
        self.assertEqual(_metadata[_mapit(gdf._data_file)], metadata)
        self.assertFalse(os.path.exists(tmppath))

    def test_put_without_metadata_support(self):
        raise unittest.SkipTest("Skip for now until metadata persistence"
                                " plans are settled")
        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(the_cont)
        self.mgr.support_metadata = False
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        self.assertEqual(gdf._obj, "z")
        self.assertEqual(gdf._obj_path, "")
        self.assertEqual(gdf._container_path,
                         os.path.join(self.td, "vol0", "bar"))
        self.assertEqual(gdf._put_datadir, the_cont)
        self.assertEqual(gdf._data_file,
                         os.path.join(self.td, "vol0", "bar", "z"))

        body = '1234\n'
        etag = md5()
        etag.update(body)
        etag = etag.hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        with gdf.create() as dw:
            self.assertIsNotNone(dw._tmppath)
            tmppath = dw._tmppath
            dw.write(body)
            dw.put(metadata)

        self.assertTrue(os.path.exists(gdf._data_file))
        self.assertEqual(_metadata, {})
        self.assertFalse(os.path.exists(tmppath))

    def test_put_ENOSPC(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._obj_path == ""
        assert gdf._container_path == os.path.join(self.td, "vol0", "bar")
        assert gdf._put_datadir == the_cont
        assert gdf._data_file == os.path.join(self.td, "vol0", "bar", "z")

        body = '1234\n'
        etag = md5()
        etag.update(body)
        etag = etag.hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        def mock_open(*args, **kwargs):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        with mock.patch("os.open", mock_open):
            try:
                with gdf.create() as dw:
                    assert dw._tmppath is not None
                    dw.write(body)
                    dw.put(metadata)
            except DiskFileNoSpace:
                pass
            else:
                self.fail("Expected exception DiskFileNoSpace")

    def test_put_rename_ENOENT(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(the_cont)
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._obj_path == ""
        assert gdf._container_path == os.path.join(self.td, "vol0", "bar")
        assert gdf._put_datadir == the_cont
        assert gdf._data_file == os.path.join(self.td, "vol0", "bar", "z")

        body = '1234\n'
        etag = md5()
        etag.update(body)
        etag = etag.hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        def mock_sleep(*args, **kwargs):
            # Return without sleep, no need to dely unit tests
            return

        def mock_rename(*args, **kwargs):
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

        with mock.patch("file_connector.swift.obj.diskfile.sleep", mock_sleep):
            with mock.patch("os.rename", mock_rename):
                try:
                    with gdf.create() as dw:
                        assert dw._tmppath is not None
                        dw.write(body)
                        dw.put(metadata)
                except OSError:
                    pass
                else:
                    self.fail("Expected exception DiskFileError")

    def test_put_obj_path(self):
        the_obj_path = os.path.join("b", "a")
        the_file = os.path.join(the_obj_path, "z")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", the_file)
        assert gdf._obj == "z"
        assert gdf._obj_path == the_obj_path
        assert gdf._container_path == os.path.join(self.td, "vol0", "bar")
        self.assertEqual(gdf._put_datadir,
                         os.path.join(self.td, "vol0", "bar", "b", "a"))
        assert gdf._data_file == os.path.join(
            self.td, "vol0", "bar", "b", "a", "z")

        body = '1234\n'
        etag = md5()
        etag.update(body)
        etag = etag.hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        with gdf.create() as dw:
            assert dw._tmppath is not None
            tmppath = dw._tmppath
            dw.write(body)
            dw.put(metadata)

        assert os.path.exists(gdf._data_file)
        assert not os.path.exists(tmppath)

    def test_delete(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")

        _mock_do_rmtree = Mock()  # Should be called
        with patch("file_connector.swift.common.utils.do_rmtree",
                   _mock_do_rmtree):
            gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
            self.assertEquals(gdf._obj, "z")
            self.assertEquals(gdf._data_file, the_file)
            later = float(gdf.read_metadata()['X-Timestamp']) + 1
            gdf.delete(normalize_timestamp(later))
            self.assertTrue(os.path.isdir(gdf._put_datadir))
            self.assertFalse(
                os.path.exists(os.path.join(gdf._put_datadir, gdf._obj)))
        self.assertTrue(_mock_do_rmtree.called)

    def test_delete_same_timestamp(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._data_file == the_file
        now = float(gdf.read_metadata()['X-Timestamp'])
        gdf.delete(normalize_timestamp(now))
        assert os.path.isdir(gdf._put_datadir)
        assert os.path.exists(os.path.join(gdf._put_datadir, gdf._obj))

    def test_delete_file_not_found(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        _mock_do_rmtree = Mock()  # Should be called
        with patch("file_connector.swift.common.utils.do_rmtree",
                   _mock_do_rmtree):
            gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
            self.assertEquals(gdf._obj, "z")
            self.assertEquals(gdf._data_file, the_file)
            later = float(gdf.read_metadata()['X-Timestamp']) + 1

            # Handle the case the file is not in the directory listing.
            os.unlink(the_file)

            gdf.delete(normalize_timestamp(later))
            self.assertTrue(os.path.isdir(gdf._put_datadir))
            self.assertFalse(
                os.path.exists(os.path.join(gdf._put_datadir, gdf._obj)))
        self.assertTrue(_mock_do_rmtree.called)

    def test_delete_file_unlink_error(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "wb") as fd:
            fd.write("1234")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        assert gdf._obj == "z"
        assert gdf._data_file == the_file

        later = float(gdf.read_metadata()['X-Timestamp']) + 1

        def _mock_os_unlink_eacces_err(f):
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))

        stats = os.stat(the_path)
        try:
            os.chmod(the_path, stats.st_mode & (~stat.S_IWUSR))

            # Handle the case os_unlink() raises an OSError
            with patch("os.unlink", _mock_os_unlink_eacces_err):
                try:
                    gdf.delete(normalize_timestamp(later))
                except OSError as e:
                    assert e.errno == errno.EACCES
                else:
                    self.fail("Excepted an OSError when unlinking file")
        finally:
            os.chmod(the_path, stats.st_mode)

        assert os.path.isdir(gdf._put_datadir)
        assert os.path.exists(os.path.join(gdf._put_datadir, gdf._obj))

    def test_delete_is_dir(self):
        the_path = os.path.join(self.td, "vol0", "bar")
        the_dir = os.path.join(the_path, "d")
        os.makedirs(the_dir)

        _mock_do_rmtree = Mock()  # Should be called
        with patch("file_connector.swift.common.utils.do_rmtree",
                   _mock_do_rmtree):
            gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "d")
            self.assertEquals(gdf._data_file, the_dir)
            later = float(gdf.read_metadata()['X-Timestamp']) + 1
            gdf.delete(normalize_timestamp(later))
            self.assertTrue(os.path.isdir(gdf._put_datadir))
            self.assertFalse(
                os.path.exists(os.path.join(gdf._put_datadir, gdf._obj)))
        self.assertTrue(_mock_do_rmtree.called)

    def test_create(self):
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir/z")
        saved_tmppath = ''
        saved_fd = None
        with gdf.create() as dw:
            self.assertEqual(gdf._put_datadir,
                             os.path.join(self.td, "vol0", "bar", "dir"))
            assert os.path.isdir(gdf._put_datadir)
            saved_tmppath = dw._tmppath
            assert os.path.dirname(saved_tmppath) == gdf._put_datadir
            assert os.path.basename(saved_tmppath)[:3] == '.z.'
            assert os.path.exists(saved_tmppath)
            dw.write("123")
            saved_fd = dw._fd
        # At the end of previous with block a close on fd is called.
        # Calling os.close on the same fd will raise an OSError
        # exception and we must catch it.
        try:
            os.close(saved_fd)
        except OSError:
            pass
        else:
            self.fail("Exception expected")
        assert not os.path.exists(saved_tmppath)

    def test_create_err_on_close(self):
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir/z")
        saved_tmppath = ''
        with gdf.create() as dw:
            self.assertEqual(gdf._put_datadir,
                             os.path.join(self.td, "vol0", "bar", "dir"))
            assert os.path.isdir(gdf._put_datadir)
            saved_tmppath = dw._tmppath
            assert os.path.dirname(saved_tmppath) == gdf._put_datadir
            assert os.path.basename(saved_tmppath)[:3] == '.z.'
            assert os.path.exists(saved_tmppath)
            dw.write("123")
            # Closing the fd prematurely should not raise any exceptions.
            dw.close()
        assert not os.path.exists(saved_tmppath)

    def test_create_err_on_unlink(self):
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "dir/z")
        saved_tmppath = ''
        with gdf.create() as dw:
            self.assertEqual(gdf._put_datadir,
                             os.path.join(self.td, "vol0", "bar", "dir"))
            assert os.path.isdir(gdf._put_datadir)
            saved_tmppath = dw._tmppath
            assert os.path.dirname(saved_tmppath) == gdf._put_datadir
            assert os.path.basename(saved_tmppath)[:3] == '.z.'
            assert os.path.exists(saved_tmppath)
            dw.write("123")
            os.unlink(saved_tmppath)
        assert not os.path.exists(saved_tmppath)

    def test_unlink_not_called_after_rename(self):
        the_obj_path = os.path.join("b", "a")
        the_file = os.path.join(the_obj_path, "z")
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", the_file)

        body = '1234\n'
        etag = md5(body).hexdigest()
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': etag,
            'Content-Length': '5',
        }

        _mock_do_unlink = Mock()  # Shouldn't be called
        with patch("file_connector.swift.obj.diskfile.do_unlink",
                   _mock_do_unlink):
            with gdf.create() as dw:
                assert dw._tmppath is not None
                tmppath = dw._tmppath
                dw.write(body)
                dw.put(metadata)
                # do_unlink is not called if dw._tmppath is set to None
                assert dw._tmppath is None
        self.assertFalse(_mock_do_unlink.called)

        assert os.path.exists(gdf._data_file)  # Real file exists
        assert not os.path.exists(tmppath)  # Temp file does not exist

    def test_fd_closed_when_diskfile_open_raises_exception_race(self):
        # do_open() succeeds but read_metadata() fails
        _m_do_open = Mock(return_value=999)
        _m_do_fstat = Mock(
            return_value=os.stat_result((33261, 2753735, 2053, 1, 1000,
                                         1000, 6873, 1431415969,
                                         1376895818, 1433139196)))
        _m_rmd = Mock(side_effect=IOError(errno.ENOENT,
                                          os.strerror(errno.ENOENT)))
        _m_do_close = Mock()
        _m_log = Mock()

        with nested(
                patch("file_connector.swift.obj.diskfile.do_open",
                      _m_do_open),
                patch("file_connector.swift.obj.diskfile.do_fstat",
                      _m_do_fstat),
                patch("file_connector.swift.obj.diskfile.do_close",
                      _m_do_close),
                patch("file_connector.swift.obj.diskfile.logging.warn",
                      _m_log)):
            gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
            gdf._mp.read_metadata = _m_rmd
            try:
                with gdf.open():
                    pass
            except DiskFileNotExist:
                pass
            else:
                self.fail("Expecting DiskFileNotExist")
            _m_do_fstat.assert_called_once_with(999)
            _m_rmd.assert_called_once_with(gdf._data_file, 999)
            _m_do_close.assert_called_once_with(999)
            self.assertFalse(gdf._fd)
            # Make sure ENOENT failure is logged
            self.assertTrue("failed with ENOENT" in _m_log.call_args[0][0])

    def test_fd_closed_when_diskfile_open_raises_DiskFileExpired(self):
        # A GET/DELETE on an expired object should close fd
        the_path = os.path.join(self.td, "vol0", "bar")
        the_file = os.path.join(the_path, "z")
        os.makedirs(the_path)
        with open(the_file, "w") as fd:
            fd.write("1234")
        md = {
            'X-Type': 'Object',
            'X-Object-Type': 'file',
            'Content-Length': str(os.path.getsize(the_file)),
            'ETag': md5("1234").hexdigest(),
            'X-Timestamp': os.stat(the_file).st_mtime,
            'X-Delete-At': 0,  # This is in the past
            'Content-Type': 'application/octet-stream'}
        _metadata[_mapit(the_file)] = md

        _m_do_close = Mock()

        with patch("file_connector.swift.obj.diskfile.do_close", _m_do_close):
            gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
            try:
                with gdf.open():
                    pass
            except DiskFileExpired:
                # Confirm that original exception is re-raised
                pass
            else:
                self.fail("Expecting DiskFileExpired")
            self.assertEqual(_m_do_close.call_count, 1)
            self.assertFalse(gdf._fd)
            # Close the actual fd, as we had mocked do_close
            os.close(_m_do_close.call_args[0][0])

    def make_directory_chown_call(self):
        path = os.path.join(self.td, "a/b/c")
        _m_do_chown = Mock()
        with patch("file_connector.swift.obj.diskfile.do_chown", _m_do_chown):
            diskfile.make_directory(path, -1, -1)
        self.assertFalse(_m_do_chown.called)
        self.assertTrue(os.path.isdir(path))

        path = os.path.join(self.td, "d/e/f")
        _m_do_chown.reset_mock()
        with patch("file_connector.swift.obj.diskfile.do_chown", _m_do_chown):
            diskfile.make_directory(path, -1, 99)
        self.assertEqual(_m_do_chown.call_count, 3)
        self.assertTrue(os.path.isdir(path))

        path = os.path.join(self.td, "g/h/i")
        _m_do_chown.reset_mock()
        with patch("file_connector.swift.obj.diskfile.do_chown", _m_do_chown):
            diskfile.make_directory(path, 99, -1)
        self.assertEqual(_m_do_chown.call_count, 3)
        self.assertTrue(os.path.isdir(path))

    def test_fchown_not_called_on_default_uid_gid_values(self):
        the_cont = os.path.join(self.td, "vol0", "bar")
        os.makedirs(the_cont)
        body = '1234'
        metadata = {
            'X-Timestamp': '1234',
            'Content-Type': 'file',
            'ETag': md5(body).hexdigest(),
            'Content-Length': len(body),
        }

        _m_do_fchown = Mock()
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47", "bar", "z")
        with gdf.create() as dw:
            assert dw._tmppath is not None
            tmppath = dw._tmppath
            dw.write(body)
            with patch("file_connector.swift.obj.diskfile.do_fchown",
                       _m_do_fchown):
                dw.put(metadata)
        self.assertFalse(_m_do_fchown.called)
        assert os.path.exists(gdf._data_file)
        assert not os.path.exists(tmppath)

    def test_unlink_not_called_on_non_existent_object(self):
        # Create container dir
        cpath = os.path.join(self.td, "vol0", "container")
        os.makedirs(cpath)
        self.assertTrue(os.path.exists(cpath))

        # This file does not exist
        obj_path = os.path.join(cpath, "object")
        self.assertFalse(os.path.exists(obj_path))

        # Create diskfile instance and check attribute initialization
        gdf = _get_diskfile(self.mgr, "vol0", "p57", "ufo47",
                            "container", "object")
        assert gdf._obj == "object"
        assert gdf._data_file == obj_path
        self.assertFalse(gdf._disk_file_does_not_exist)

        # Simulate disk file call sequence issued during DELETE request.
        # And confirm that read_metadata() and unlink() is not called.
        self.assertRaises(DiskFileNotExist, gdf.read_metadata)
        self.assertTrue(gdf._disk_file_does_not_exist)
        _m_rmd = Mock()
        _m_do_unlink = Mock()
        gdf._mp.read_metadata = _m_rmd
        with patch("file_connector.swift.obj.diskfile.do_unlink",
                   _m_do_unlink):
            gdf.delete(0)
        self.assertFalse(_m_rmd.called)
        self.assertFalse(_m_do_unlink.called)

    def _system_can_zero_copy(self):
        if not splice.available:
            return False

        try:
            get_md5_socket()
        except IOError:
            return False

        return True

    def test_zero_copy_cache_dropping(self):
        if not self._system_can_zero_copy():
            raise unittest.SkipTest("zero-copy support is missing")
        self.mgr.pipe_size = 8

        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "ufo47", "bar", "z", fsize=163840)
        with gdf.open():
            reader = gdf.reader()

        self.assertTrue(reader.can_zero_copy_send())
        with mock.patch("file_connector.swift.obj.diskfile.DiskFileReader."
                        "_drop_cache") as dbc:
            with mock.patch("file_connector.swift.obj.diskfile."
                            "DROP_CACHE_WINDOW", 4095):
                with open('/dev/null', 'w') as devnull:
                    reader.zero_copy_send(devnull.fileno())
                expected = (4 * 10) + 1
                self.assertEqual(len(dbc.mock_calls), expected)

    def test_tee_to_md5_pipe_length_mismatch(self):
        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "ufo47", "bar", "z", fsize=16385)
        with gdf.open():
            reader = gdf.reader()
        self.assertTrue(reader.can_zero_copy_send())

        with mock.patch('file_connector.swift.obj.diskfile.tee') as mock_tee:
            mock_tee.side_effect = lambda _1, _2, _3, cnt: cnt - 1

            with open('/dev/null', 'w') as devnull:
                exc_re = (r'tee\(\) failed: tried to move \d+ bytes, but only '
                          'moved -?\d+')
                try:
                    reader.zero_copy_send(devnull.fileno())
                except Exception as e:
                    self.assertTrue(re.match(exc_re, str(e)))
                else:
                    self.fail('Expected Exception was not raised')

    def test_splice_to_wsockfd_blocks(self):
        gdf = _create_and_get_diskfile(self.mgr, self.td, "vol0", "p57",
                                       "ufo47", "bar", "z", fsize=16385)
        with gdf.open():
            reader = gdf.reader()
        self.assertTrue(reader.can_zero_copy_send())

        def _run_test():
            # Set up mock of `splice`
            splice_called = [False]  # State hack

            def fake_splice(fd_in, off_in, fd_out, off_out, len_, flags):
                if fd_out == devnull.fileno() and not splice_called[0]:
                    splice_called[0] = True
                    err = errno.EWOULDBLOCK
                    raise IOError(err, os.strerror(err))

                return splice(fd_in, off_in, fd_out, off_out,
                              len_, flags)

            mock_splice.side_effect = fake_splice

            # Set up mock of `trampoline`
            # There are 2 reasons to mock this:
            #
            # - We want to ensure it's called with the expected arguments at
            #   least once
            # - When called with our write FD (which points to `/dev/null`), we
            #   can't actually call `trampoline`, because adding such FD to an
            #   `epoll` handle results in `EPERM`
            def fake_trampoline(fd, read=None, write=None, timeout=None,
                                timeout_exc=timeout.Timeout,
                                mark_as_closed=None):
                if write and fd == devnull.fileno():
                    return
                else:
                    hubs.trampoline(fd, read=read, write=write,
                                    timeout=timeout, timeout_exc=timeout_exc,
                                    mark_as_closed=mark_as_closed)

            mock_trampoline.side_effect = fake_trampoline

            reader.zero_copy_send(devnull.fileno())

            # Assert the end of `zero_copy_send` was reached
            self.assertTrue(mock_close.called)
            # Assert there was at least one call to `trampoline` waiting for
            # `write` access to the output FD
            mock_trampoline.assert_any_call(devnull.fileno(), write=True)
            # Assert at least one call to `splice` with the output FD we expect
            for call in mock_splice.call_args_list:
                args = call[0]
                if args[2] == devnull.fileno():
                    break
            else:
                self.fail('`splice` not called with expected arguments')

        with mock.patch('file_connector.swift.obj.diskfile.splice') as \
                mock_splice:
            with mock.patch.object(
                    reader, 'close', side_effect=reader.close) as mock_close:
                with open('/dev/null', 'w') as devnull:
                    with mock.patch('file_connector.swift.obj.diskfile.'
                                    'trampoline') as mock_trampoline:
                        _run_test()
