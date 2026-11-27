"""Current-user DPAPI credential storage outside the checkout. Never log secrets."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def crypt(payload, decrypt=False):
    if os.name != 'nt':
        raise RuntimeError('Windows current-user DPAPI required')
    buffer = ctypes.create_string_buffer(payload)
    source = Blob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    if decrypt:
        ok = ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
    else:
        ok = ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), 'image2ppt providers', None, None, None, 1, ctypes.byref(target))
    if not ok:
        raise RuntimeError('DPAPI operation failed for current user')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


def location():
    return Path(os.environ['LOCALAPPDATA']) / 'image2ppt' / 'credentials' / 'providers.dpapi'


def load(provider):
    value = json.loads(crypt(location().read_bytes(), decrypt=True)).get(provider)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError('Provider credential missing: ' + provider)
    return value


def install(data):
    if set(data) != {'moma', 'opencode_go'} or any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError('Exactly two nonempty provider credentials required')
    destination = location()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Restrict the directory before writing; DPAPI additionally binds the user.
    account = subprocess.check_output(['whoami'], text=True).strip()
    subprocess.run(['icacls', str(destination.parent), '/inheritance:r', '/grant:r', account + ':(OI)(CI)F', 'SYSTEM:(OI)(CI)F'], check=True, capture_output=True)
    if destination.exists():
        raise FileExistsError('Credential store already exists; explicit rotation required')
    destination.write_bytes(crypt(json.dumps(data).encode('utf8')))
    assert all(load(k) == v for k, v in data.items())
    return {'status': 'configured', 'providers': sorted(data), 'encrypted': True,
            'scope': 'Windows current user', 'opencode_go_invocation': 'NOT_TESTED'}


def rotate(data):
    if set(data) != {'moma'} or not isinstance(data['moma'],str) or not data['moma'].strip():
        raise ValueError('Only explicit MoMA rotation supported')
    destination=location()
    old=json.loads(crypt(destination.read_bytes(),decrypt=True))
    old.update(data)
    temporary=destination.with_suffix('.new.dpapi')
    temporary.write_bytes(crypt(json.dumps(old).encode('utf8')))
    temporary.replace(destination)
    assert load('moma')==data['moma']
    return {'status':'rotated','provider':'moma','encrypted':True}


if __name__ == '__main__':
    if sys.argv[1:] == ['install-stdin']:
        print(json.dumps(install(json.load(sys.stdin))))
    elif sys.argv[1:] == ['rotate-moma-stdin']:
        print(json.dumps(rotate(json.load(sys.stdin))))
    elif sys.argv[1:] == ['status']:
        print(json.dumps({k: bool(load(k)) for k in ('moma', 'opencode_go')}))
    else:
        raise SystemExit('Use install-stdin or status; no credential arguments accepted')
