"""Download the OFL fonts the layered pipeline matches against.

Binaries stay in fonts/layered/ and are gitignored. Official Adobe release
zips are large; only the Simplified Chinese Regular/Bold cuts are kept.
PowerPoint and the preview rasterizer only draw a family that is installed.
Fitting may still use these files, and the pipeline then emits the closest
installed family.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'fonts'/'layered'

# SHA256 of the extracted OTF, filled after the first verified download.
PINNED = {
    'SourceHanSansSC-Regular.otf': 'f1d8611151880c6c336aabeac4640ef434fa13cbfbf1ffe82d0a71b2a5637256',
    'SourceHanSansSC-Bold.otf': 'df2b90f5bcc6d01dfc964cec5f6d535d6b6aebd26ed7fd79a9c1b3f2112fcb6b',
    'SourceHanSerifSC-Regular.otf': '78aa7a328fd974df2d688c8a9fd74a33d8334dfa84ab24d9d11efb2ffc464117',
}

PACKAGES = [
    {'url': 'https://github.com/adobe-fonts/source-han-sans/releases/download/2.005R/09_SourceHanSansSC.zip',
     'members': {
         'SourceHanSansSC-Regular.otf': {'family': 'Source Han Sans SC', 'bold': False},
         'SourceHanSansSC-Bold.otf': {'family': 'Source Han Sans SC', 'bold': True},
     }},
    {'url': 'https://github.com/adobe-fonts/source-han-serif/releases/download/2.003R/09_SourceHanSerifSC.zip',
     'members': {
         'SourceHanSerifSC-Regular.otf': {'family': 'Source Han Serif SC', 'bold': False},
     }},
]


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'image2ppt-layered-fonts'})
    print('downloading', url, flush=True)
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _extract(blob, name):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        match = [info for info in archive.infolist() if info.filename.endswith('/'+name) or info.filename == name]
        if len(match) != 1:
            raise RuntimeError(f'{name} not found once in archive ({len(match)})')
        data = archive.read(match[0])
    if not data.startswith(b'OTTO'):
        raise RuntimeError(name+' is not an OpenType font')
    return data


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for package in PACKAGES:
        pending = []
        for name, meta in package['members'].items():
            path = DEST/name
            expected = PINNED[name]
            if path.is_file() and expected and _digest(path.read_bytes()) == expected:
                records.append({'path': name, 'family': meta['family'], 'bold': meta['bold'],
                                'index': 0, 'sha256': expected})
                print(name, 'cached', expected, flush=True)
                continue
            pending.append((name, meta))
        if not pending:
            continue
        blob = _download(package['url'])
        for name, meta in pending:
            data = _extract(blob, name)
            digest = _digest(data)
            expected = PINNED[name]
            if expected and digest != expected:
                raise SystemExit(f'{name} sha256 {digest} != {expected}')
            (DEST/name).write_bytes(data)
            records.append({'path': name, 'family': meta['family'], 'bold': meta['bold'],
                            'index': 0, 'sha256': digest})
            print(name, digest, len(data), flush=True)
    manifest = {
        'fonts': records,
        'note': 'OFL Source Han cuts for layered ink fitting. Not embedded in the PPTX. '
                'Install the same files if PowerPoint should draw these family names; '
                'otherwise the pipeline emits the closest font already installed, usually 微软雅黑.',
    }
    (DEST/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('wrote', DEST/'manifest.json')
    print('Install the .otf files only if previews should use Source Han names. '
          'Fitting works without installing them.')


if __name__ == '__main__':
    sys.exit(main() or 0)
