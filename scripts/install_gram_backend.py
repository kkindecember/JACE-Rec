"""Fetch the pinned GRAM source and verify the installed runtime files."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INITIALIZERS = {'__init__.py': '', 'model/__init__.py': '',
                'processor/__init__.py': 'from .Collator import CollatorGRAM\n'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(name, raw):
    text = raw.decode('utf-8').replace('from IPython import embed\n', '')
    if name == 'model/gram.py':
        text = text.replace('.view(', '.reshape(')
    return text.encode('utf-8')


def install(source=None):
    manifest = json.loads((ROOT/'assets/gram-backend-manifest.json').read_text())
    expected = {name: row['installed_sha256'] for name, row in manifest['files'].items()}
    expected.update({name: digest(text.encode()) for name, text in INITIALIZERS.items()})
    destination = ROOT/'vendor/gram'
    if destination.exists():
        for name, sha in expected.items():
            path = destination/name
            if not path.is_file() or digest(path.read_bytes()) != sha:
                raise FileExistsError('Different backend already installed; move it aside first: '+str(path))
        print('Verified existing GRAM backend:', manifest['revision'])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gram-download-', dir=destination.parent) as temp:
        stage = Path(temp)/'gram'
        for name, row in manifest['files'].items():
            if source is None:
                request = Request(row['url'], headers={'User-Agent': 'JACE-Rec'})
                with urlopen(request, timeout=60) as response:
                    raw = response.read()
            else:
                raw = (source/name).read_bytes()
            if digest(raw) != row['download_sha256']:
                raise ValueError('Upstream source checksum mismatch: '+name)
            installed = transform(name, raw)
            if digest(installed) != row['installed_sha256']:
                raise ValueError('Installed source checksum mismatch: '+name)
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(installed)
        for name, text in INITIALIZERS.items():
            (stage/name).write_text(text)
        stage.rename(destination)
    print('Installed GRAM backend:', manifest['revision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='source', type=Path,
                        help='Optional src directory of an unmodified pinned GRAM checkout')
    args = parser.parse_args()
    install(args.source)


if __name__ == '__main__':
    main()
