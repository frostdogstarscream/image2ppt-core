"""Host-wide named, cross-process capacity limits (also usable by GPU services)."""
from contextlib import contextmanager
import os
from pathlib import Path
import time
import tempfile

CAPACITY={'page':2,'cpu':2,'moma':2,'go':1,'office':1,'compiler':1,'gpu':1}


@contextmanager
def lease(name,root=None,capacity=None,timeout=3600):
    name=name.lower()
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):raise ValueError('invalid resource')
    root=Path(root or os.environ.get('IMAGE2PPT_RESOURCE_ROOT') or Path(tempfile.gettempdir())/'image2ppt-resource-locks')
    root.mkdir(parents=True,exist_ok=True)
    limit=capacity or CAPACITY.get(name.split('-')[0],1);deadline=time.monotonic()+timeout
    handle=None
    while handle is None:
        for slot in range(limit):
            f=(root/f'{name}-{slot}.lock').open('a+b')
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                handle=f;break
            except (OSError,BlockingIOError):f.close()
        if handle is None:
            if time.monotonic()>deadline:raise TimeoutError('resource wait exceeded: '+name)
            time.sleep(.1)
    try:yield
    finally:
        if os.name=='nt':
            import msvcrt
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        handle.close()
