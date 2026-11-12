"""Trusted container entrypoint. Install an irreversible syscall filter first."""
import ctypes
import ctypes.util
import errno
import json
import runpy
import sys


def restrict_processes():
    library=ctypes.util.find_library('seccomp')
    if not library:
        raise RuntimeError('libseccomp is required by the image-script sandbox')
    seccomp=ctypes.CDLL(library,use_errno=True)
    seccomp.seccomp_init.argtypes=[ctypes.c_uint32]
    seccomp.seccomp_init.restype=ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype=ctypes.c_int
    seccomp.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    seccomp.seccomp_load.argtypes=[ctypes.c_void_p]
    seccomp.seccomp_release.argtypes=[ctypes.c_void_p]
    context=seccomp.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError('seccomp filter allocation failed')
    try:
        for name in ('execve','execveat','fork','vfork','clone','clone3','socket','socketpair'):
            number=seccomp.seccomp_syscall_resolve_name(name.encode())
            if number>=0 and seccomp.seccomp_rule_add(context,0x00050000|errno.EPERM,number,0)!=0:
                raise RuntimeError('seccomp rule installation failed')
        if seccomp.seccomp_load(context)!=0:
            raise RuntimeError('seccomp load failed')
    finally:
        seccomp.seccomp_release(context)


if __name__=='__main__':
    try:
        # Load supported libraries before disallowing further thread creation.
        import PIL
        import numpy
        import cv2
        import resource
        cv2.setNumThreads(1)
        resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024*1024,32*1024*1024))
        print('SANDBOX_ENVIRONMENT '+json.dumps({'python':sys.version.split()[0],
            'Pillow':PIL.__version__,'numpy':numpy.__version__,'opencv':cv2.__version__}),flush=True)
        restrict_processes()
    except Exception as exc:
        print('SANDBOX_ISOLATION_UNAVAILABLE '+type(exc).__name__,file=sys.stderr)
        sys.exit(78)
    runpy.run_path('/input/script.py',run_name='__main__')
