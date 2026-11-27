"""Download the official ModelScope snapshot, binding every file to its SHA-256."""
import concurrent.futures, hashlib, json, pathlib, time, urllib.request, urllib.parse

ROOT = pathlib.Path('/data/workspaces/qwen-image-layered-20261104')
API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-Layered/repo/files?Revision=master&Recursive=true'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def fetch(row):
    dest = ROOT/'models'/row['Path']; dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == row['Size'] and digest(dest) == row['Sha256']: return
    part = dest.with_name(dest.name+'.partial')
    url = 'https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-Layered/repo?'+urllib.parse.urlencode({'Revision':row['Revision'], 'FilePath':row['Path']})
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req, timeout=120) as response:
                append = offset and response.status == 206
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'): raise ValueError('bad range')
                with part.open('ab' if append else 'wb') as stream:
                    for chunk in iter(lambda: response.read(4*1024*1024), b''): stream.write(chunk)
            if part.stat().st_size != row['Size'] or digest(part) != row['Sha256']: raise ValueError('size/hash mismatch')
            part.rename(dest); print('VERIFIED', row['Path'], row['Size'], flush=True); return
        except Exception as error:
            print('RETRY', row['Path'], attempt, str(error), flush=True)
            if attempt == 2: raise
            time.sleep(2)

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT/'model-manifest.json'
    if manifest.exists(): rows = json.loads(manifest.read_text())
    else:
        data = json.load(urllib.request.urlopen(API, timeout=30))
        rows = [r for r in data['Data']['Files'] if r.get('Size',0)>0]
        manifest.write_text(json.dumps(rows, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: list(executor.map(fetch, rows))
    (ROOT/'download-complete.json').write_text(json.dumps({'files':len(rows),'bytes':sum(r['Size'] for r in rows),'sha256_verified':True}))
