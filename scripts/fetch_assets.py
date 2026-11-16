"""Restore verified JACE release assets and pinned upstream encoders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('Missing or corrupt asset: '+str(path))


def download(url, destination, record):
    if destination.exists():
        verify(destination, record)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record['bytes']:
        raise ValueError('Oversized partial download: '+str(partial))
    if offset < record['bytes']:
        headers = {'User-Agent': 'JACE-Rec'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        with urlopen(Request(url, headers=headers), timeout=120) as response:
            if offset and response.status == 206:
                content_range = response.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {offset}-'):
                    raise ValueError('Incorrect HTTP range response')
                mode = 'ab'
            else:
                mode = 'wb'
            with partial.open(mode) as stream:
                shutil.copyfileobj(response, stream, length=8 << 20)
    verify(partial, record)
    partial.replace(destination)
    return destination


def extract(archive_path, root, expected):
    """Extract only manifest-listed regular files, with per-file checksums."""
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if (name not in expected or name in seen or not member.isfile()
                    or Path(name).is_absolute() or '..' in Path(name).parts):
                raise ValueError('Unexpected archive member: '+name)
            record = expected[name]
            if member.size != record['bytes']:
                raise ValueError('Unexpected asset size: '+name)
            destination = root/name
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError('Asset destination escapes the checkout: '+name)
            seen.add(name)
            if destination.exists():
                verify(destination, record)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with archive.extractfile(member) as source:
                        shutil.copyfileobj(source, stream, length=8 << 20)
                    stream.flush()
                    verify(temporary, record)
                    # Do not overwrite an asset created by a concurrent restore.
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError('Archive is missing declared files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['toys', 'beauty', 'sports', 'yelp', 'all'], default='toys')
    parser.add_argument('--repository', help='GitHub OWNER/REPOSITORY containing release assets')
    parser.add_argument('--from', dest='source', type=Path, help='Local directory containing the release archives')
    parser.add_argument('--t5-source', type=Path, help='Optional local copy of the pinned T5 files')
    parser.add_argument('--encoder', action='store_true', help='Download only the optional Qwen item encoder')
    args = parser.parse_args()
    if args.encoder:
        encoder = json.loads((ROOT/'assets/encoder-manifest.json').read_text())
        for name, record in encoder['files'].items():
            download(record['url'], ROOT/name, record)
        return
    distribution = json.loads((ROOT/'assets/distribution.json').read_text())
    if args.source is None and not args.repository:
        parser.error('Specify --repository OWNER/REPO or --from /path/to/release-assets')
    files = json.loads((ROOT/'assets/manifest.json').read_text())['files']
    files.update(json.loads((ROOT/'assets/training-manifest.json').read_text())['files'])
    domains = list(distribution['bundles']) if args.domain == 'all' else [args.domain]
    for domain in domains:
        record = distribution['bundles'][domain]
        if args.source:
            archive = args.source/record['file']
            verify(archive, record)
        else:
            url = f'https://github.com/{args.repository}/releases/download/{distribution["release_tag"]}/{record["file"]}'
            archive = download(url, ROOT/'.cache/downloads'/record['file'], record)
        extract(archive, ROOT, {name: files[name] for name in record['files']})
        print('Restored dataset and three seeds:', domain)
    for name, record in distribution['upstream'].items():
        if args.t5_source:
            source = args.t5_source/Path(name).name
            verify(source, record)
            destination = ROOT/name
            if destination.exists():
                verify(destination, record)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        else:
            download(record['url'], ROOT/name, record)
    print('All requested assets verified')


if __name__ == '__main__':
    main()
