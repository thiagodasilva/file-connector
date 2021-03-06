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

""" Tests for common.utils """

import os
import json
import unittest
import errno
import xattr
import tempfile
import hashlib
import tarfile
import shutil
import cPickle as pickle
from collections import defaultdict
from mock import patch, mock_open, Mock
from file_connector.swift.common import utils
from file_connector.swift.common.utils import deserialize_metadata, \
    serialize_metadata, PICKLE_PROTOCOL, XattrMetadataPersistence, \
    JsonMetadataPersistence
from file_connector.swift.common.exceptions import FileConnectorFileSystemOSError,\
    FileConnectorFileSystemIOError
from file_connector.swift.common.fs_utils import mkdirs
from swift.common.exceptions import DiskFileNoSpace
from test.unit import DATA_DIR

try:
    import scandir
    scandir_present = True
except ImportError:
    scandir_present = False

#
# Somewhat hacky way of emulating the operation of xattr calls. They are made
# against a dictionary that stores the xattr key/value pairs.
#
_xattrs = {}
_xattr_op_cnt = defaultdict(int)
_xattr_set_err = {}
_xattr_get_err = {}
_xattr_rem_err = {}
_xattr_set = None
_xattr_get = None
_xattr_remove = None


def _xkey(path, key):
    return "%s:%s" % (path, key)


def _setxattr(path, key, value, *args, **kwargs):
    _xattr_op_cnt['set'] += 1
    xkey = _xkey(path, key)
    if xkey in _xattr_set_err:
        e = IOError()
        e.errno = _xattr_set_err[xkey]
        raise e
    global _xattrs
    _xattrs[xkey] = value


def _getxattr(path, key, *args, **kwargs):
    _xattr_op_cnt['get'] += 1
    xkey = _xkey(path, key)
    if xkey in _xattr_get_err:
        e = IOError()
        e.errno = _xattr_get_err[xkey]
        raise e
    global _xattrs
    if xkey in _xattrs:
        ret_val = _xattrs[xkey]
    else:
        e = IOError("Fake IOError")
        e.errno = errno.ENODATA
        raise e
    return ret_val


def _removexattr(path, key, *args, **kwargs):
    _xattr_op_cnt['remove'] += 1
    xkey = _xkey(path, key)
    if xkey in _xattr_rem_err:
        e = IOError()
        e.errno = _xattr_rem_err[xkey]
        raise e
    global _xattrs
    if xkey in _xattrs:
        del _xattrs[xkey]
    else:
        e = IOError("Fake IOError")
        e.errno = errno.ENODATA
        raise e


def _initxattr():
    global _xattrs
    _xattrs = {}
    global _xattr_op_cnt
    _xattr_op_cnt = defaultdict(int)
    global _xattr_set_err, _xattr_get_err, _xattr_rem_err
    _xattr_set_err = {}
    _xattr_get_err = {}
    _xattr_rem_err = {}

    # Save the current methods
    global _xattr_set;    _xattr_set    = xattr.setxattr
    global _xattr_get;    _xattr_get    = xattr.getxattr
    global _xattr_remove; _xattr_remove = xattr.removexattr

    # Monkey patch the calls we use with our internal unit test versions
    xattr.setxattr    = _setxattr
    xattr.getxattr    = _getxattr
    xattr.removexattr = _removexattr


def _destroyxattr():
    # Restore the current methods just in case
    global _xattr_set;    xattr.setxattr    = _xattr_set
    global _xattr_get;    xattr.getxattr    = _xattr_get
    global _xattr_remove; xattr.removexattr = _xattr_remove
    # Destroy the stored values and
    global _xattrs; _xattrs = None


class SimMemcache(object):
    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key, None)

    def set(self, key, value):
        self._d[key] = value


def _mock_os_fsync(fd):
    return


class TestSafeUnpickler(unittest.TestCase):

    class Exploit(object):
        def __reduce__(self):
            return (os.system, ('touch /tmp/pickle-exploit',))

    def test_loads(self):
        valid_md = {'key1': 'val1', 'key2': 'val2'}
        for protocol in (0, 1, 2):
            valid_dump = pickle.dumps(valid_md, protocol)
            mal_dump = pickle.dumps(self.Exploit(), protocol)
            # malicious dump is appended to valid dump
            payload1 = valid_dump[:-1] + mal_dump
            # malicious dump is prefixed to valid dump
            payload2 = mal_dump[:-1] + valid_dump
            # entire dump is malicious
            payload3 = mal_dump
            for payload in (payload1, payload2, payload3):
                try:
                    utils.SafeUnpickler.loads(payload)
                except pickle.UnpicklingError as err:
                    self.assertTrue('Potentially unsafe pickle' in err)
                else:
                    self.fail("Expecting cPickle.UnpicklingError")


class TestJsonMetadataPersistence(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.mp = JsonMetadataPersistence(self.td)

    def tearDown(self):
        shutil.rmtree(self.td)

    def test_get_metadata_dir_obj(self):
        path = os.path.join(self.td, 'cont/obj')
        expected_dir = os.path.join(self.td, 'cont', self.mp.meta_dir, 'obj')
        expected_file = os.path.join(self.td, 'cont', self.mp.meta_dir, 'obj',
                                     self.mp.obj_meta)
        dir_path, file_path = self.mp._get_metadata_dir(path)

        self.assertEqual(expected_dir, dir_path)
        self.assertEqual(expected_file, file_path)

        # subdirs sanity
        path = os.path.join(self.td, 'cont/subdir1/subdir2/obj')
        expected_dir = os.path.join(self.td, 'cont/subdir1/subdir2',
                                    self.mp.meta_dir, 'obj')
        expected_file = os.path.join(self.td, 'cont/subdir1/subdir2',
                                     self.mp.meta_dir, 'obj',
                                     self.mp.obj_meta)
        dir_path, file_path = self.mp._get_metadata_dir(path)

        self.assertEqual(expected_dir, dir_path)
        self.assertEqual(expected_file, file_path)

    def test_get_metadata_dir_cont(self):
        path = os.path.join(self.td, 'cont')
        expected_dir = os.path.join(self.td, 'cont', self.mp.meta_dir)
        expected_file = os.path.join(self.td, 'cont', self.mp.meta_dir,
                                     self.mp.cont_meta)
        dir_path, file_path = self.mp._get_metadata_dir(path)

        self.assertEqual(expected_dir, dir_path)
        self.assertEqual(expected_file, file_path)

    def test_write_metadata(self):
        path = os.path.join(self.td, 'cont/obj')
        md_file = os.path.join(self.td, 'cont', self.mp.meta_dir,
                               'obj', self.mp.obj_meta)
        orig_d = {'bar': 'foo'}
        self.mp.write_metadata(path, orig_d)
        try:
            with open(md_file, 'rt') as f:
                md = utils.deserialize_metadata(f.read())
        except Exception as e:
            self.fail(e)

        expected_md = {'bar': 'foo'}
        self.assertEqual(expected_md, md)

    def test_write_metadata_etag(self):
        # don't store fake etag
        path = "/tmp/foo/cont/obj"
        md_file = os.path.join('/tmp/foo/cont', self.mp.meta_dir,
                               'obj', self.mp.obj_meta)
        orig_d = {'bar': 'foo', 'ETag': 'fcetag'}
        self.mp.write_metadata(path, orig_d)
        try:
            with open(md_file, 'rt') as f:
                md = utils.deserialize_metadata(f.read())
        except Exception as e:
            self.fail(e)

        # but store a real etag
        expected_md = {'bar': 'foo'}
        self.assertEqual(expected_md, md)

        orig_d = {'bar': 'foo', 'ETag': 'realetag'}
        self.mp.write_metadata(path, orig_d)
        try:
            with open(md_file, 'rt') as f:
                md = utils.deserialize_metadata(f.read())
        except Exception as e:
            self.fail(e)

        expected_md = {'bar': 'foo', 'ETag': 'realetag'}
        self.assertEqual(expected_md, md)

    def test_write_metadata_write_fail_nospace(self):
        write_mock = Mock(side_effect=IOError(errno.ENOSPC, "fail"))
        path = "/tmp/foo/cont/obj"
        orig_d = {'bar': 'foo'}
        mo = mock_open()
        with patch('__builtin__.open', mo):
            mock_file = mo.return_value
            mock_file.write = write_mock
            self.assertRaises(
                DiskFileNoSpace,
                self.mp.write_metadata, path, orig_d)

    def test_write_metadata_fail(self):
        write_mock = Mock(side_effect=IOError(errno.EIO, "fail"))
        path = os.path.join(self.td, 'cont/obj')
        orig_d = {'bar': 'foo'}
        mo = mock_open()
        with patch('__builtin__.open', mo):
            mock_file = mo.return_value
            mock_file.write = write_mock
            self.assertRaises(
                FileConnectorFileSystemIOError,
                self.mp.write_metadata, path, orig_d)

    def test_write_metadata_rename_fail(self):
        rename_mock = Mock(side_effect=OSError('failed rename'))
        path = os.path.join(self.td, 'cont/obj')
        orig_d = {'bar': 'foo'}
        with patch('os.rename', rename_mock):
            self.assertRaises(
                FileConnectorFileSystemOSError,
                self.mp.write_metadata, path, orig_d)

    def test_read_obj_metadata(self):
        path = os.path.join(self.td, 'cont/obj')
        md_dir = os.path.join(self.td, 'cont', self.mp.meta_dir, 'obj')
        md_file = os.path.join(self.td, 'cont', self.mp.meta_dir,
                               'obj', self.mp.obj_meta)
        orig_d = {'bar': 'foo'}
        mkdirs(md_dir)
        with open(md_file, 'wt') as f:
            f.write(serialize_metadata(orig_d))

        md = self.mp.read_metadata(path)
        self.assertEqual(orig_d, md)

    def test_read_cont_metadata(self):
        path = os.path.join(self.td, 'cont')
        md_dir = os.path.join(self.td, 'cont', self.mp.meta_dir)
        md_file = os.path.join(self.td, 'cont', self.mp.meta_dir,
                               self.mp.cont_meta)
        orig_d = {'bar': 'foo'}
        mkdirs(md_dir)
        with open(md_file, 'wt') as f:
            f.write(serialize_metadata(orig_d))

        md = self.mp.read_metadata(path)
        self.assertEqual(orig_d, md)

    def test_read_metadata_nofile(self):
        path = os.path.join(self.td, 'cont/obj')
        md = self.mp.read_metadata(path)
        self.assertEqual({}, md)

    def test_read_metadata_fail(self):
        read_mock = Mock(side_effect=IOError(errno.EIO, "fail"))
        path = os.path.join(self.td, 'cont/obj')
        mo = mock_open()
        with patch('__builtin__.open', mo):
            mock_file = mo.return_value
            mock_file.read = read_mock
            self.assertRaises(
                IOError,
                self.mp.read_metadata, path)


class TestUtils(unittest.TestCase):
    """ Tests for common.utils """

    def setUp(self):
        _initxattr()
        self.mp = XattrMetadataPersistence('/tmp/foo')

    def tearDown(self):
        _destroyxattr()

    def test_write_metadata(self):
        path = "/tmp/foo/w"
        orig_d = {'bar': 'foo'}
        self.mp.write_metadata(path, orig_d)
        xkey = _xkey(path, utils.METADATA_KEY)
        self.assertEqual(1, len(_xattrs))
        self.assertIn(xkey, _xattrs)
        self.assertEqual(orig_d, deserialize_metadata(_xattrs[xkey]))
        self.assertEqual(_xattr_op_cnt['set'], 1)

    def test_write_metadata_err(self):
        path = "/tmp/foo/w"
        orig_d = {'bar': 'foo'}
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattr_set_err[xkey] = errno.EOPNOTSUPP
        try:
            self.mp.write_metadata(path, orig_d)
        except IOError as e:
            assert e.errno == errno.EOPNOTSUPP
            assert len(_xattrs.keys()) == 0
            assert _xattr_op_cnt['set'] == 1
        else:
            self.fail("Expected an IOError exception on write")

    def test_write_metadata_space_err(self):

        def _mock_xattr_setattr(item, name, value):
            raise IOError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        with patch('xattr.setxattr', _mock_xattr_setattr):
            path = "/tmp/foo/w"
            orig_d = {'bar': 'foo'}
            try:
                self.mp.write_metadata(path, orig_d)
            except DiskFileNoSpace:
                pass
            else:
                self.fail("Expected DiskFileNoSpace exception")
            fd = 0
            try:
                self.mp.write_metadata(fd, orig_d)
            except DiskFileNoSpace:
                pass
            else:
                self.fail("Expected DiskFileNoSpace exception")

    def test_write_metadata_multiple(self):
        # At 64 KB an xattr key/value pair, this should generate three keys.
        path = "/tmp/foo/w"
        orig_d = {'bar': 'x' * 150000}
        self.mp.write_metadata(path, orig_d)
        self.assertEqual(len(_xattrs.keys()), 3,
                         "Expected 3 keys, found %d" % len(_xattrs.keys()))
        payload = ''
        for i in range(0, 3):
            xkey = _xkey(path, "%s%s" % (utils.METADATA_KEY, i or ''))
            assert xkey in _xattrs
            assert len(_xattrs[xkey]) <= utils.MAX_XATTR_SIZE
            payload += _xattrs[xkey]
        assert orig_d == deserialize_metadata(payload)
        assert _xattr_op_cnt['set'] == 3, "%r" % _xattr_op_cnt

    def test_clean_metadata(self):
        path = "/tmp/foo/c"
        expected_d = {'a': 'y' * 150000}
        expected_p = serialize_metadata(expected_d)
        for i in range(0, 3):
            xkey = _xkey(path, "%s%s" % (utils.METADATA_KEY, i or ''))
            _xattrs[xkey] = expected_p[:utils.MAX_XATTR_SIZE]
            expected_p = expected_p[utils.MAX_XATTR_SIZE:]
        assert not expected_p
        self.mp._clean_metadata(path)
        assert _xattr_op_cnt['remove'] == 4, "%r" % _xattr_op_cnt

    def test_clean_metadata_err(self):
        path = "/tmp/foo/c"
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattrs[xkey] = serialize_metadata({'a': 'y'})
        _xattr_rem_err[xkey] = errno.EOPNOTSUPP
        try:
            self.mp._clean_metadata(path)
        except IOError as e:
            assert e.errno == errno.EOPNOTSUPP
            assert _xattr_op_cnt['remove'] == 1, "%r" % _xattr_op_cnt
        else:
            self.fail("Expected an IOError exception on remove")

    def test_read_metadata(self):
        path = "/tmp/foo/r"
        expected_d = {'a': 'y'}
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattrs[xkey] = serialize_metadata(expected_d)
        res_d = self.mp.read_metadata(path)
        assert res_d == expected_d, "Expected %r, result %r" % \
            (expected_d, res_d)
        assert _xattr_op_cnt['get'] == 1, "%r" % _xattr_op_cnt

    def test_read_metadata_notfound(self):
        path = "/tmp/foo/r"
        res_d = self.mp.read_metadata(path)
        assert res_d == {}
        assert _xattr_op_cnt['get'] == 1, "%r" % _xattr_op_cnt

    def test_read_metadata_err(self):
        path = "/tmp/foo/r"
        expected_d = {'a': 'y'}
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattrs[xkey] = serialize_metadata(expected_d)
        _xattr_get_err[xkey] = errno.EOPNOTSUPP
        try:
            self.mp.read_metadata(path)
        except IOError as e:
            assert e.errno == errno.EOPNOTSUPP
            assert (_xattr_op_cnt['get'] == 1), "%r" % _xattr_op_cnt
        else:
            self.fail("Expected an IOError exception on get")

    def test_read_metadata_multiple(self):
        path = "/tmp/foo/r"
        expected_d = {'a': 'y' * 150000}
        expected_p = serialize_metadata(expected_d)
        for i in range(0, 3):
            xkey = _xkey(path, "%s%s" % (utils.METADATA_KEY, i or ''))
            _xattrs[xkey] = expected_p[:utils.MAX_XATTR_SIZE]
            expected_p = expected_p[utils.MAX_XATTR_SIZE:]
        assert not expected_p
        res_d = self.mp.read_metadata(path)
        assert res_d == expected_d, "Expected %r, result %r" % \
            (expected_d, res_d)
        assert _xattr_op_cnt['get'] == 4, "%r" % _xattr_op_cnt

    def test_read_metadata_multiple_one_missing(self):
        path = "/tmp/foo/r"
        expected_d = {'a': 'y' * 150000}
        expected_p = serialize_metadata(expected_d)
        for i in range(0, 2):
            xkey = _xkey(path, "%s%s" % (utils.METADATA_KEY, i or ''))
            _xattrs[xkey] = expected_p[:utils.MAX_XATTR_SIZE]
            expected_p = expected_p[utils.MAX_XATTR_SIZE:]
        assert len(expected_p) <= utils.MAX_XATTR_SIZE
        res_d = self.mp.read_metadata(path)
        assert res_d == {}
        assert _xattr_op_cnt['get'] == 3, "%r" % _xattr_op_cnt

    def test_restore_metadata_none(self):
        # No initial metadata
        path = "/tmp/foo/i"
        res_d = self.mp.restore_metadata(path, {'b': 'y'}, {})
        expected_d = {'b': 'y'}
        self.assertEqual(
            res_d, expected_d,
            "Expected %r, result %r" % (expected_d, res_d))
        self.assertEqual(_xattr_op_cnt['set'], 1, "%r" % _xattr_op_cnt)

    def test_restore_metadata(self):
        # Initial metadata
        path = "/tmp/foo/i"
        initial_d = {'a': 'z'}
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattrs[xkey] = serialize_metadata(initial_d)
        res_d = self.mp.restore_metadata(path, {'b': 'y'}, initial_d)
        expected_d = {'a': 'z', 'b': 'y'}
        self.assertEqual(
            res_d, expected_d,
            "Expected %r, result %r" % (expected_d, res_d))
        assert _xattr_op_cnt['set'] == 1, "%r" % _xattr_op_cnt

    def test_restore_metadata_nochange(self):
        # Initial metadata but no changes
        path = "/tmp/foo/i"
        initial_d = {'a': 'z'}
        xkey = _xkey(path, utils.METADATA_KEY)
        _xattrs[xkey] = serialize_metadata(initial_d)
        res_d = self.mp.restore_metadata(path, {}, initial_d)
        expected_d = {'a': 'z'}
        self.assertEqual(res_d, expected_d,
                         "Expected %r, result %r" % (expected_d, res_d))
        assert _xattr_op_cnt['set'] == 0, "%r" % _xattr_op_cnt

    def test_deserialize_metadata_pickle(self):
        orig__read_pickled_metadata = utils._read_pickled_metadata
        orig_md = {'key1': 'value1', 'key2': 'value2'}
        pickled_md = pickle.dumps(orig_md, PICKLE_PROTOCOL)
        _m_pickle_loads = Mock(return_value={})
        try:
            with patch('file_connector.swift.common.utils.pickle.loads',
                       _m_pickle_loads):
                # Conf option turned off
                utils._read_pickled_metadata = False
                # pickled
                utils.deserialize_metadata(pickled_md)
                self.assertFalse(_m_pickle_loads.called)
                _m_pickle_loads.reset_mock()
                # not pickled
                utils.deserialize_metadata("not_pickle")
                self.assertFalse(_m_pickle_loads.called)
                _m_pickle_loads.reset_mock()

                # Conf option turned on
                utils._read_pickled_metadata = True
                # pickled
                md = utils.deserialize_metadata(pickled_md)
                self.assertTrue(_m_pickle_loads.called)
                self.assertTrue(isinstance(md, dict))
                _m_pickle_loads.reset_mock()
                # not pickled
                utils.deserialize_metadata("not_pickle")
                self.assertFalse(_m_pickle_loads.called)
                _m_pickle_loads.reset_mock()

                # malformed pickle
                _m_pickle_loads.side_effect = pickle.UnpicklingError
                md = utils.deserialize_metadata("malformed_pickle")
                self.assertTrue(isinstance(md, dict))
        finally:
            utils._read_pickled_metadata = orig__read_pickled_metadata

    def test_deserialize_metadata_json(self):
        orig_md = {'key1': 'value1', 'key2': 'value2'}
        json_md = json.dumps(orig_md, separators=(',', ':'))
        _m_json_loads = Mock(return_value={})
        with patch('file_connector.swift.common.utils.json.loads',
                   _m_json_loads):
            utils.deserialize_metadata("not_json")
            self.assertFalse(_m_json_loads.called)
            _m_json_loads.reset_mock()
            utils.deserialize_metadata("{fake_valid_json}")
            self.assertTrue(_m_json_loads.called)
            _m_json_loads.reset_mock()

    def test_add_timestamp_empty(self):
        orig = {}
        res = utils._add_timestamp(orig)
        assert res == {}

    def test_add_timestamp_none(self):
        orig = {'a': 1, 'b': 2, 'c': 3}
        exp = {'a': (1, 0), 'b': (2, 0), 'c': (3, 0)}
        res = utils._add_timestamp(orig)
        assert res == exp

    def test_add_timestamp_mixed(self):
        orig = {'a': 1, 'b': (2, 1), 'c': 3}
        exp = {'a': (1, 0), 'b': (2, 1), 'c': (3, 0)}
        res = utils._add_timestamp(orig)
        assert res == exp

    def test_add_timestamp_all(self):
        orig = {'a': (1, 0), 'b': (2, 1), 'c': (3, 0)}
        res = utils._add_timestamp(orig)
        assert res == orig

    def test_get_etag_empty(self):
        tf = tempfile.NamedTemporaryFile()
        hd = utils._get_etag(tf.name)
        assert hd == hashlib.md5().hexdigest()

    def test_get_etag(self):
        tf = tempfile.NamedTemporaryFile()
        tf.file.write('123' * utils.CHUNK_SIZE)
        tf.file.flush()
        hd = utils._get_etag(tf.name)
        tf.file.seek(0)
        md5 = hashlib.md5()
        while True:
            chunk = tf.file.read(utils.CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)
        assert hd == md5.hexdigest()

    def test_get_etag_dup_fd_closed(self):
        fd, path = tempfile.mkstemp()
        data = "It's not who we are underneath, but what we do that defines us"
        os.write(fd, data)
        os.lseek(fd, 0, os.SEEK_SET)

        mock_do_close = Mock()
        with patch("file_connector.swift.common.utils.do_close",
                   mock_do_close):
            etag = utils._get_etag(fd)
        self.assertEqual(etag, hashlib.md5(data).hexdigest())
        self.assertTrue(mock_do_close.called)

        # We mocked out close, so we have to close the fd for real
        os.close(mock_do_close.call_args[0][0])
        os.close(fd)

    def test_get_object_metadata_dne(self):
        md = self.mp.get_object_metadata("/tmp/doesNotEx1st")
        assert md == {}

    def test_get_object_metadata_err(self):
        tf = tempfile.NamedTemporaryFile()
        try:
            self.mp.get_object_metadata(
                os.path.join(tf.name, "doesNotEx1st"))
        except FileConnectorFileSystemOSError as e:
            assert e.errno != errno.ENOENT
        else:
            self.fail("Expected exception")

    obj_keys = (utils.X_TIMESTAMP, utils.X_CONTENT_TYPE, utils.X_ETAG,
                utils.X_CONTENT_LENGTH, utils.X_TYPE, utils.X_OBJECT_TYPE)

    def test_get_object_metadata_file(self):
        tf = tempfile.NamedTemporaryFile()
        tf.file.write('123')
        tf.file.flush()
        md = self.mp.get_object_metadata(tf.name)
        for key in self.obj_keys:
            assert key in md, "Expected key %s in %r" % (key, md)
        assert md[utils.X_TYPE] == utils.OBJECT
        assert md[utils.X_OBJECT_TYPE] == utils.FILE
        assert md[utils.X_CONTENT_TYPE] == utils.FILE_TYPE
        assert md[utils.X_CONTENT_LENGTH] == os.path.getsize(tf.name)
        assert md[utils.X_TIMESTAMP] == utils.normalize_timestamp(
            os.path.getctime(tf.name))
        assert md[utils.X_ETAG] == utils._get_etag(tf.name)

    def test_get_object_metadata_dir(self):
        td = tempfile.mkdtemp()
        try:
            md = self.mp.get_object_metadata(td)
            for key in self.obj_keys:
                assert key in md, "Expected key %s in %r" % (key, md)
            assert md[utils.X_TYPE] == utils.OBJECT
            assert md[utils.X_OBJECT_TYPE] == utils.DIR_NON_OBJECT
            assert md[utils.X_CONTENT_TYPE] == utils.DIR_TYPE
            assert md[utils.X_CONTENT_LENGTH] == 0
            self.assertEqual(md[utils.X_TIMESTAMP],
                             utils.normalize_timestamp(os.path.getctime(td)))
            assert md[utils.X_ETAG] == hashlib.md5().hexdigest()
        finally:
            os.rmdir(td)

    def test_create_object_metadata_file(self):
        tf = tempfile.NamedTemporaryFile()
        tf.file.write('4567')
        tf.file.flush()
        r_md = self.mp.create_object_metadata(tf.name)

        xkey = _xkey(tf.name, utils.METADATA_KEY)
        assert len(_xattrs.keys()) == 1
        assert xkey in _xattrs
        assert _xattr_op_cnt['set'] == 1
        md = deserialize_metadata(_xattrs[xkey])
        self.assertEqual(r_md, md)

        for key in self.obj_keys:
            assert key in md, "Expected key %s in %r" % (key, md)
        assert md[utils.X_TYPE] == utils.OBJECT
        assert md[utils.X_OBJECT_TYPE] == utils.FILE
        assert md[utils.X_CONTENT_TYPE] == utils.FILE_TYPE
        assert md[utils.X_CONTENT_LENGTH] == os.path.getsize(tf.name)
        assert md[utils.X_TIMESTAMP] == utils.normalize_timestamp(
            os.path.getctime(tf.name))
        assert md[utils.X_ETAG] == utils._get_etag(tf.name)

    def test_create_object_metadata_dir(self):
        td = tempfile.mkdtemp()
        try:
            r_md = self.mp.create_object_metadata(td)

            xkey = _xkey(td, utils.METADATA_KEY)
            assert len(_xattrs.keys()) == 1
            assert xkey in _xattrs
            assert _xattr_op_cnt['set'] == 1
            md = deserialize_metadata(_xattrs[xkey])
            assert r_md == md

            for key in self.obj_keys:
                assert key in md, "Expected key %s in %r" % (key, md)
            assert md[utils.X_TYPE] == utils.OBJECT
            assert md[utils.X_OBJECT_TYPE] == utils.DIR_NON_OBJECT
            assert md[utils.X_CONTENT_TYPE] == utils.DIR_TYPE
            assert md[utils.X_CONTENT_LENGTH] == 0
            assert md[utils.X_TIMESTAMP] == utils.normalize_timestamp(
                os.path.getctime(td))
            assert md[utils.X_ETAG] == hashlib.md5().hexdigest()
        finally:
            os.rmdir(td)

    def test_get_container_metadata(self):
        def _mock_get_container_details(path, mp):
            o_list = ['a', 'b', 'c']
            o_count = 3
            b_used = 47
            return o_list, o_count, b_used
        orig_gcd = utils.get_container_details
        utils.get_container_details = _mock_get_container_details
        td = tempfile.mkdtemp()
        try:
            exp_md = {
                utils.X_TYPE: (utils.CONTAINER, 0),
                utils.X_TIMESTAMP: (utils.normalize_timestamp(
                    os.path.getctime(td)), 0),
                utils.X_PUT_TIMESTAMP: (utils.normalize_timestamp(
                    os.path.getmtime(td)), 0),
                utils.X_OBJECTS_COUNT: (3, 0),
                utils.X_BYTES_USED: (47, 0),
            }
            md = self.mp.get_container_metadata(td)
            assert md == exp_md
        finally:
            utils.get_container_details = orig_gcd
            os.rmdir(td)

    def test_get_account_metadata(self):
        def _mock_get_account_details(path):
            c_list = ['123', 'abc']
            c_count = 2
            return c_list, c_count
        orig_gad = utils.get_account_details
        utils.get_account_details = _mock_get_account_details
        td = tempfile.mkdtemp()
        try:
            exp_md = {
                utils.X_TYPE: (utils.ACCOUNT, 0),
                utils.X_TIMESTAMP: (utils.normalize_timestamp(
                    os.path.getctime(td)), 0),
                utils.X_PUT_TIMESTAMP: (utils.normalize_timestamp(
                    os.path.getmtime(td)), 0),
                utils.X_OBJECTS_COUNT: (0, 0),
                utils.X_BYTES_USED: (0, 0),
                utils.X_CONTAINER_COUNT: (2, 0),
            }
            md = utils.get_account_metadata(td)
            assert md == exp_md
        finally:
            utils.get_account_details = orig_gad
            os.rmdir(td)

    cont_keys = [utils.X_TYPE, utils.X_TIMESTAMP, utils.X_PUT_TIMESTAMP,
                 utils.X_OBJECTS_COUNT, utils.X_BYTES_USED]

    def test_create_container_metadata(self):
        td = tempfile.mkdtemp()
        try:
            r_md = self.mp.create_container_metadata(td)

            xkey = _xkey(td, utils.METADATA_KEY)
            assert len(_xattrs.keys()) == 1
            assert xkey in _xattrs
            assert _xattr_op_cnt['get'] == 1
            assert _xattr_op_cnt['set'] == 1
            md = deserialize_metadata(_xattrs[xkey])
            assert r_md == md

            for key in self.cont_keys:
                assert key in md, "Expected key %s in %r" % (key, md)
            assert md[utils.X_TYPE] == (utils.CONTAINER, 0)
            assert md[utils.X_TIMESTAMP] == (
                utils.normalize_timestamp(os.path.getctime(td)), 0)
            assert md[utils.X_PUT_TIMESTAMP] == (utils.normalize_timestamp(
                os.path.getmtime(td)), 0)
            assert md[utils.X_OBJECTS_COUNT] == (0, 0)
            assert md[utils.X_BYTES_USED] == (0, 0)
        finally:
            os.rmdir(td)

    acct_keys = [val for val in cont_keys]
    acct_keys.append(utils.X_CONTAINER_COUNT)

    def test_create_account_metadata(self):
        td = tempfile.mkdtemp()
        try:
            r_md = self.mp.create_account_metadata(td)

            xkey = _xkey(td, utils.METADATA_KEY)
            assert len(_xattrs.keys()) == 1
            assert xkey in _xattrs
            assert _xattr_op_cnt['get'] == 1
            assert _xattr_op_cnt['set'] == 1
            md = deserialize_metadata(_xattrs[xkey])
            assert r_md == md

            for key in self.acct_keys:
                assert key in md, "Expected key %s in %r" % (key, md)
            assert md[utils.X_TYPE] == (utils.ACCOUNT, 0)
            assert md[utils.X_TIMESTAMP] == (utils.normalize_timestamp(
                os.path.getctime(td)), 0)
            assert md[utils.X_PUT_TIMESTAMP] == (utils.normalize_timestamp(
                os.path.getmtime(td)), 0)
            assert md[utils.X_OBJECTS_COUNT] == (0, 0)
            assert md[utils.X_BYTES_USED] == (0, 0)
            assert md[utils.X_CONTAINER_COUNT] == (0, 0)
        finally:
            os.rmdir(td)

    def test_get_account_details(self):
        orig_cwd = os.getcwd()
        td = tempfile.mkdtemp()
        try:
            swiftdir = os.path.join(DATA_DIR, "account_tree.tar.bz2")
            tf = tarfile.open(swiftdir, "r:bz2")
            os.chdir(td)
            tf.extractall()

            container_list, container_count = utils.get_account_details(td)
            assert container_count == 3
            assert set(container_list) == set(['c1', 'c2', 'c3'])
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(td)

    def test_get_account_details_notadir(self):
        tf = tempfile.NamedTemporaryFile()
        try:
            utils.get_account_details(tf.name)
        except OSError as err:
            if err.errno != errno.ENOTDIR:
                self.fail("Expecting ENOTDIR")
        else:
            self.fail("Expecting ENOTDIR")

    def test_get_container_details_notadir(self):
        tf = tempfile.NamedTemporaryFile()
        obj_list, object_count, bytes_used = \
            utils.get_container_details(tf.name, self.mp)
        assert bytes_used == 0
        assert object_count == 0
        assert obj_list == []

    def test_get_container_details(self):
        orig_cwd = os.getcwd()
        __do_getsize = utils._do_getsize
        td = tempfile.mkdtemp()
        try:
            swiftdir = os.path.join(DATA_DIR, "container_tree.tar.bz2")
            tf = tarfile.open(swiftdir, "r:bz2")
            os.chdir(td)
            tf.extractall()

            utils._do_getsize = False

            obj_list, object_count, bytes_used = \
                utils.get_container_details(td, self.mp)
            assert bytes_used == 0, repr(bytes_used)
            # Should not include the directories
            assert object_count == 5, repr(object_count)
            assert set(obj_list) == set(['file1', 'file3', 'file2',
                                         'dir1/file1', 'dir1/file2'
                                         ]), repr(obj_list)
        finally:
            utils._do_getsize = __do_getsize
            os.chdir(orig_cwd)
            shutil.rmtree(td)

    def test_get_container_details_from_fs_do_getsize_true(self):
        orig_cwd = os.getcwd()
        __do_getsize = utils._do_getsize
        td = tempfile.mkdtemp()
        try:
            swiftdir = os.path.join(DATA_DIR, "container_tree.tar.bz2")
            tf = tarfile.open(swiftdir, "r:bz2")
            os.chdir(td)
            tf.extractall()

            utils._do_getsize = True

            obj_list, object_count, bytes_used = \
                utils.get_container_details(td, self.mp)
            assert bytes_used == 30, repr(bytes_used)
            assert object_count == 5, repr(object_count)
            assert set(obj_list) == set(['file1', 'file3', 'file2',
                                         'dir1/file1', 'dir1/file2'
                                         ]), repr(obj_list)
        finally:
            utils._do_getsize = __do_getsize
            os.chdir(orig_cwd)
            shutil.rmtree(td)

    def test_validate_container_empty(self):
        ret = utils.validate_container({})
        assert not ret

    def test_validate_container_missing_keys(self):
        ret = utils.validate_container({'foo': 'bar'})
        assert not ret

    def test_validate_container_bad_type(self):
        md = {utils.X_TYPE: ('bad', 0),
              utils.X_TIMESTAMP: ('na', 0),
              utils.X_PUT_TIMESTAMP: ('na', 0),
              utils.X_OBJECTS_COUNT: ('na', 0),
              utils.X_BYTES_USED: ('na', 0)}
        ret = utils.validate_container(md)
        assert not ret

    def test_validate_container_good_type(self):
        md = {utils.X_TYPE: (utils.CONTAINER, 0),
              utils.X_TIMESTAMP: ('na', 0),
              utils.X_PUT_TIMESTAMP: ('na', 0),
              utils.X_OBJECTS_COUNT: ('na', 0),
              utils.X_BYTES_USED: ('na', 0)}
        ret = utils.validate_container(md)
        assert ret

    def test_validate_account_empty(self):
        ret = utils.validate_account({})
        assert not ret

    def test_validate_account_missing_keys(self):
        ret = utils.validate_account({'foo': 'bar'})
        assert not ret

    def test_validate_account_bad_type(self):
        md = {utils.X_TYPE: ('bad', 0),
              utils.X_TIMESTAMP: ('na', 0),
              utils.X_PUT_TIMESTAMP: ('na', 0),
              utils.X_OBJECTS_COUNT: ('na', 0),
              utils.X_BYTES_USED: ('na', 0),
              utils.X_CONTAINER_COUNT: ('na', 0)}
        ret = utils.validate_account(md)
        assert not ret

    def test_validate_account_good_type(self):
        md = {utils.X_TYPE: (utils.ACCOUNT, 0),
              utils.X_TIMESTAMP: ('na', 0),
              utils.X_PUT_TIMESTAMP: ('na', 0),
              utils.X_OBJECTS_COUNT: ('na', 0),
              utils.X_BYTES_USED: ('na', 0),
              utils.X_CONTAINER_COUNT: ('na', 0)}
        ret = utils.validate_account(md)
        assert ret

    def test_validate_object_empty(self):
        ret = utils.validate_object({})
        assert not ret

    def test_validate_object_missing_keys(self):
        ret = utils.validate_object({'foo': 'bar'})
        assert not ret

    def test_validate_object_bad_type(self):
        md = {utils.X_TIMESTAMP: 'na',
              utils.X_CONTENT_TYPE: 'na',
              utils.X_ETAG: 'bad',
              utils.X_CONTENT_LENGTH: 'na',
              utils.X_TYPE: 'bad',
              utils.X_OBJECT_TYPE: 'na'}
        ret = utils.validate_object(md)
        assert not ret

    def test_validate_object_good_type(self):
        md = {utils.X_TIMESTAMP: 'na',
              utils.X_CONTENT_TYPE: 'na',
              utils.X_ETAG: 'bad',
              utils.X_CONTENT_LENGTH: 'na',
              utils.X_TYPE: utils.OBJECT,
              utils.X_OBJECT_TYPE: 'na'}
        ret = utils.validate_object(md)
        assert ret

    def test_validate_object_with_stat(self):
        md = {utils.X_TIMESTAMP: 'na',
              utils.X_CONTENT_TYPE: 'na',
              utils.X_ETAG: 'bad',
              utils.X_CONTENT_LENGTH: '12345',
              utils.X_TYPE: utils.OBJECT,
              utils.X_OBJECT_TYPE: 'na'}
        fake_stat = Mock(st_size=12346, st_mode=33188)
        self.assertFalse(utils.validate_object(md, fake_stat))
        fake_stat = Mock(st_size=12345, st_mode=33188)
        self.assertTrue(utils.validate_object(md, fake_stat))

    def test_validate_object_marker_dir(self):
        md = {utils.X_TIMESTAMP: 'na',
              utils.X_CONTENT_TYPE: 'application/directory',
              utils.X_ETAG: 'bad',
              utils.X_CONTENT_LENGTH: '0',
              utils.X_TYPE: utils.OBJECT,
              utils.X_OBJECT_TYPE: utils.DIR_OBJECT}
        fake_stat = Mock(st_size=4096, st_mode=16744)
        self.assertTrue(utils.validate_object(md, fake_stat))


class TestUtilsDirObjects(unittest.TestCase):

    def setUp(self):
        _initxattr()
        self.dirs = [
            'dir1',
            'dir1/dir2',
            'dir1/dir2/dir3']
        self.files = [
            'file1',
            'file2',
            'dir1/dir2/file3']
        self.tempdir = tempfile.mkdtemp()
        self.rootdir = os.path.join(self.tempdir, 'a')
        self.mp = XattrMetadataPersistence(self.rootdir)
        for d in self.dirs:
            os.makedirs(os.path.join(self.rootdir, d))
        for f in self.files:
            open(os.path.join(self.rootdir, f), 'w').close()

    def tearDown(self):
        _destroyxattr()
        shutil.rmtree(self.tempdir)

    def _set_dir_object(self, obj):
        metadata = self.mp.read_metadata(os.path.join(self.rootdir, obj))
        metadata[utils.X_OBJECT_TYPE] = utils.DIR_OBJECT
        self.mp.write_metadata(os.path.join(self.rootdir, self.dirs[0]),
                               metadata)

    def _clear_dir_object(self, obj):
        metadata = self.mp.read_metadata(os.path.join(self.rootdir, obj))
        metadata[utils.X_OBJECT_TYPE] = utils.DIR_NON_OBJECT
        self.mp.write_metadata(os.path.join(self.rootdir, obj),
                               metadata)

    def test_rmobjdir_removing_files(self):
        self.assertFalse(utils.rmobjdir(self.mp, self.rootdir))

        # Remove the files
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        self.assertTrue(utils.rmobjdir(self.mp, self.rootdir))

    def test_rmobjdir_removing_dirs(self):
        self.assertFalse(utils.rmobjdir(self.mp, self.rootdir))

        # Remove the files
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        self._set_dir_object(self.dirs[0])
        self.assertFalse(utils.rmobjdir(self.mp, self.rootdir))
        self._clear_dir_object(self.dirs[0])
        self.assertTrue(utils.rmobjdir(self.mp, self.rootdir))

    def test_rmobjdir_metadata_errors(self):

        def _mock_rm(path):
            print "_mock_rm-metadata_errors(%s)" % path
            if path.endswith("dir3"):
                raise OSError(13, "foo")
            return {}

        _orig_rm = self.mp.read_metadata
        self.mp.read_metadata = _mock_rm
        try:
            try:
                utils.rmobjdir(self.mp, self.rootdir)
            except OSError:
                pass
            else:
                self.fail("Expected OSError")
        finally:
            self.mp.read_metadata = _orig_rm

    def test_rmobjdir_metadata_enoent(self):

        def _mock_rm(path):
            print "_mock_rm-metadata_enoent(%s)" % path
            shutil.rmtree(path)
            raise FileConnectorFileSystemIOError(errno.ENOENT,
                                                 os.strerror(errno.ENOENT))

        # Remove the files
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        _orig_rm = self.mp.read_metadata
        self.mp.read_metadata = _mock_rm
        try:
            try:
                self.assertTrue(utils.rmobjdir(self.mp, self.rootdir))
            except IOError:
                self.fail("Unexpected IOError")
            else:
                pass
        finally:
            self.mp.read_metadata = _orig_rm

    def test_rmobjdir_rmdir_enoent(self):

        seen = [0]
        _orig_rm = utils.do_rmdir

        def _mock_rm(path):
            print "_mock_rm-rmdir_enoent(%s)" % path
            if path == self.rootdir and not seen[0]:
                seen[0] = 1
                raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY))
            else:
                shutil.rmtree(path)
                raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

        # Remove the files
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        utils.do_rmdir = _mock_rm
        try:
            try:
                self.assertTrue(utils.rmobjdir(self.mp, self.rootdir))
            except OSError:
                self.fail("Unexpected OSError")
            else:
                pass
        finally:
            utils.do_rmdir = _orig_rm

    def test_rmobjdir_rmdir_error(self):

        seen = [0]
        _orig_rm = utils.do_rmdir

        def _mock_rm(path):
            print "_mock_rm-rmdir_enoent(%s)" % path
            if path == self.rootdir and not seen[0]:
                seen[0] = 1
                raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY))
            else:
                raise OSError(errno.EACCES, os.strerror(errno.EACCES))

        # Remove the files
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        utils.do_rmdir = _mock_rm
        try:
            try:
                utils.rmobjdir(self.mp, self.rootdir)
            except OSError:
                pass
            else:
                self.fail("Expected OSError")
        finally:
            utils.do_rmdir = _orig_rm

    def test_rmobjdir_files_left_in_top_dir(self):

        seen = [0]
        _orig_rm = utils.do_rmdir

        def _mock_rm(path):
            print "_mock_rm-files_left_in_top_dir(%s)" % path
            if path == self.rootdir:
                if not seen[0]:
                    seen[0] = 1
                    raise OSError(errno.ENOTEMPTY,
                                  os.strerror(errno.ENOTEMPTY))
                else:
                    return _orig_rm(path)
            else:
                shutil.rmtree(path)
                raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

        # Remove the files, leaving the ones at the root
        for f in self.files:
            if f.startswith('dir'):
                os.unlink(os.path.join(self.rootdir, f))

        utils.do_rmdir = _mock_rm
        try:
            try:
                self.assertFalse(utils.rmobjdir(self.mp, self.rootdir))
            except OSError:
                self.fail("Unexpected OSError")
            else:
                pass
        finally:
            utils.do_rmdir = _orig_rm

    def test_rmobjdir_error_final_rmdir(self):

        seen = [0]
        _orig_rm = utils.do_rmdir

        def _mock_rm(path):
            print "_mock_rm-files_left_in_top_dir(%s)" % path
            if path == self.rootdir:
                if not seen[0]:
                    seen[0] = 1
                    raise OSError(errno.ENOTEMPTY,
                                  os.strerror(errno.ENOTEMPTY))
                else:
                    raise OSError(errno.EACCES, os.strerror(errno.EACCES))
            else:
                shutil.rmtree(path)
                raise OSError(errno.ENOENT, os.strerror(errno.ENOENT))

        # Remove the files, leaving the ones at the root
        for f in self.files:
            os.unlink(os.path.join(self.rootdir, f))

        utils.do_rmdir = _mock_rm
        try:
            try:
                utils.rmobjdir(self.mp, self.rootdir)
            except OSError:
                pass
            else:
                self.fail("Expected OSError")
        finally:
            utils.do_rmdir = _orig_rm

    def test_gf_listdir(self):
        for entry in utils.gf_listdir(self.rootdir):
            if scandir_present:
                self.assertFalse(isinstance(entry, utils.SmallDirEntry))
            else:
                self.assertTrue(isinstance(entry, utils.SmallDirEntry))
            if entry.name in ('dir1'):
                self.assertTrue(entry.is_dir())
                if not scandir_present:
                    self.assertEqual(entry._d_type, utils.DT_UNKNOWN)
            elif entry.name in ('file1', 'file2'):
                self.assertFalse(entry.is_dir())


class TestSmallDirEntry(unittest.TestCase):

    def test_does_stat_when_no_d_type(self):
        e = utils.SmallDirEntry('/root/path', 'name', utils.DT_UNKNOWN)
        mock_os_lstat = Mock(return_value=Mock(st_mode=16744))
        with patch('os.lstat', mock_os_lstat):
            self.assertTrue(e.is_dir())
            self.assertTrue(e._stat)  # Make sure stat gets populated
        mock_os_lstat.assert_called_once_with('/root/path/name')

        # Subsequent calls to is_dir() should not call os.lstat()
        mock_os_lstat.reset_mock()
        with patch('os.lstat', mock_os_lstat):
            self.assertTrue(e._stat)  # Make sure stat is already populated
            self.assertTrue(e.is_dir())
        self.assertFalse(mock_os_lstat.called)

    def test_is_dir_file_not_present_should_return_false(self):
        e = utils.SmallDirEntry('/root/path', 'name', utils.DT_UNKNOWN)
        mock_os_lstat = Mock(side_effect=OSError(errno.ENOENT,
                                                 os.strerror(errno.ENOENT)))
        with patch('os.lstat', mock_os_lstat):
            self.assertFalse(e.is_dir())
