"""Compile inside the fixed compiler workspace, then copy verified artifacts to a job."""
from pathlib import Path
import shutil
from uuid import uuid4

from ..pipeline import ROOT, compile_pages


def compile_local(irs, output, texts):
    output=Path(output).resolve()
    if output.is_relative_to(ROOT):
        return compile_pages(irs,output,texts)
    # The first-party finalizer requires final files within the compiler root.
    # External job roots are supported by copying the unchanged verified package.
    scratch=ROOT/'outputs/oc'/uuid4().hex[:12]
    artifacts=compile_pages(irs,scratch,texts)
    shutil.copytree(scratch,output)
    def relocate(value):
        if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate(v) for v in value]
        if isinstance(value,(str,Path)):
            try:
                path=Path(value)
                if path.is_absolute() and path.is_relative_to(scratch):return str(output/path.relative_to(scratch))
            except (OSError,ValueError):pass
        return value
    return relocate(artifacts)
