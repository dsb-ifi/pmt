"""Download paper datasets with published checksums; never accept unverified bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads(Path(__file__).with_name('data_sources.json').read_text())


def digest(path: Path, algorithm: str = 'sha256') -> str:
    """Hash a local file in bounded chunks."""
    hasher = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def require(path: Path, expected: str, algorithm: str = 'sha256') -> None:
    """Reject missing or corrupted data instead of using it for an experiment."""
    if not path.is_file() or digest(path, algorithm) != expected:
        raise ValueError(f'{path}: {algorithm} checksum mismatch (expected {expected})')


def fetch(url: str, target: Path, expected: str | None = None, algorithm: str = 'sha256') -> None:
    """Download atomically and verify a checksum before replacing the target."""
    if target.exists() and (expected is None or digest(target, algorithm) == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'pmt-release-data'})
    with urllib.request.urlopen(request) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle, 1 << 20)
    if expected is not None:
        require(partial, expected, algorithm)
    partial.replace(target)


def electricity(root: Path) -> None:
    """Apply the exact source preprocessing with the approved optional pandas dependency."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'Electricity preprocessing requires optional pandas; alternatively use --electricity-csv'
        ) from exc
    archive = root / 'download/cache/electricityloaddiagrams20112014.zip'
    fetch(SOURCES['ELECTRICITY_ZIP'], archive)
    with zipfile.ZipFile(archive) as bundle, bundle.open('LD2011_2014.txt') as raw:
        frame = pd.read_csv(raw, parse_dates=True, sep=';', decimal=',', index_col=0)
    frame = frame.resample('1h', closed='right').sum()
    frame = frame.loc[:, frame.cumsum(axis=0).iloc[8920] != 0]
    frame.index = frame.index.rename('date')
    frame = frame['2012':]
    target = root / 'electricity/electricity.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    frame.to_csv(partial)
    require(partial, SOURCES['ELECTRICITY_SHA256'])
    partial.replace(target)


def download(name: str, root: Path, password: str | None = None, electricity_csv: Path | None = None) -> None:
    """Download and verify one dataset family from the reference's source manifest."""
    if name in SOURCES['TSTCC']:
        spec = SOURCES['TSTCC'][name]
        for filename, (file_id, checksum) in spec['files'].items():
            fetch(SOURCES['DATAVERSE'].format(file_id), root / spec['folder'] / filename, checksum, 'md5')
    elif name == 'etth1':
        fetch(SOURCES['ETTH1'], root / 'ETT-small/ETTh1.csv', SOURCES['ETTH1_SHA256'])
    elif name == 'ucr':
        missing = [
            rel
            for rel, checksum in SOURCES['UCR_SHA256'].items()
            if not (root / 'UCR' / rel).exists() or digest(root / 'UCR' / rel) != checksum
        ]
        if missing:
            if not password:
                raise ValueError('UCR requires --ucr-password from the archive briefing')
            archive = root / 'download/cache/UCRArchive_2018.zip'
            fetch(SOURCES['UCR_ZIP'], archive)
            with zipfile.ZipFile(archive) as bundle:
                for rel in missing:
                    target = root / 'UCR' / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + '.partial')
                    with bundle.open(f'UCRArchive_2018/{rel}', pwd=password.encode()) as source, partial.open(
                        'wb'
                    ) as out:
                        shutil.copyfileobj(source, out, 1 << 20)
                    require(partial, SOURCES['UCR_SHA256'][rel])
                    partial.replace(target)
    elif name == 'electricity':
        target = root / 'electricity/electricity.csv'
        if electricity_csv is not None:
            require(electricity_csv, SOURCES['ELECTRICITY_SHA256'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if electricity_csv.resolve() != target.resolve():
                shutil.copyfile(electricity_csv, target)
        elif not target.exists() or digest(target) != SOURCES['ELECTRICITY_SHA256']:
            electricity(root)
        require(target, SOURCES['ELECTRICITY_SHA256'])
    else:
        raise ValueError(f'Unknown dataset {name!r}')


def main() -> None:
    """Download selected datasets, optionally reusing the verified Electricity CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--only',
        nargs='+',
        choices=('har', 'epilepsy', 'ucr', 'etth1', 'electricity'),
        default=['har', 'epilepsy', 'ucr', 'etth1', 'electricity'],
    )
    parser.add_argument('--root', type=Path, default=ROOT / 'data')
    parser.add_argument('--ucr-password')
    parser.add_argument('--electricity-csv', type=Path)
    args = parser.parse_args()
    for name in args.only:
        download(name, args.root, args.ucr_password, args.electricity_csv)
        print(f'{name}: verified')


if __name__ == '__main__':
    main()
