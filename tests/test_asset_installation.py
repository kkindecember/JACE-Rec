import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fetch_assets', ROOT/'scripts/fetch_assets.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def record(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def archive(path, name, data, symlink=False):
    with tarfile.open(path, 'w:gz') as stream:
        info = tarfile.TarInfo(name)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = '/tmp/outside'
            stream.addfile(info)
        else:
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


class RestoreTests(unittest.TestCase):
    def test_extract_hashes_and_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'checkout'; root.mkdir()
            pack = Path(tmp)/'data.tar.gz'
            name = 'assets/data/Toys/example.json'
            data = b'{"value": 3}\n'
            archive(pack, name, data)
            fetch.extract(pack, root, {name: record(data)})
            fetch.extract(pack, root, {name: record(data)})
            self.assertEqual((root/name).read_bytes(), data)
            (root/name).write_bytes(b'existing different data')
            with self.assertRaises(ValueError):
                fetch.extract(pack, root, {name: record(data)})
            self.assertEqual((root/name).read_bytes(), b'existing different data')

    def test_rejects_traversal_symlinks_and_corrupt_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'checkout'; root.mkdir()
            pack = Path(tmp)/'data.tar.gz'
            for name, linked in [('../outside', False), ('assets/file', True)]:
                archive(pack, name, b'abc', symlink=linked)
                with self.assertRaises(ValueError):
                    fetch.extract(pack, root, {name: record(b'abc')})
            archive(pack, 'assets/file', b'bad')
            with self.assertRaises(ValueError):
                fetch.extract(pack, root, {'assets/file': record(b'abc')})
            self.assertFalse((root/'assets/file').exists())

    def test_http_resume_and_server_ignoring_range(self):
        class Response(io.BytesIO):
            def __init__(self, data, status, headers):
                super().__init__(data); self.status = status; self.headers = headers
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'file'
            partial = target.with_name('file.part')
            partial.write_bytes(b'abc')
            with patch.object(fetch, 'urlopen', return_value=Response(b'def', 206, {'Content-Range': 'bytes 3-5/6'})) as request:
                fetch.download('https://example.invalid/file', target, record(b'abcdef'))
                self.assertEqual(request.call_args[0][0].get_header('Range'), 'bytes=3-')
            self.assertEqual(target.read_bytes(), b'abcdef')
            other = Path(tmp)/'other'
            other.with_name('other.part').write_bytes(b'abc')
            with patch.object(fetch, 'urlopen', return_value=Response(b'abcdef', 200, {})):
                fetch.download('https://example.invalid/file', other, record(b'abcdef'))
            self.assertEqual(other.read_bytes(), b'abcdef')

    def test_bad_range_does_not_append(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 0-2/6'}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'file'
            partial = target.with_name('file.part'); partial.write_bytes(b'abc')
            with patch.object(fetch, 'urlopen', return_value=Response(b'abc')):
                with self.assertRaises(ValueError):
                    fetch.download('https://example.invalid/file', target, record(b'abcdef'))
            self.assertEqual(partial.read_bytes(), b'abc')
