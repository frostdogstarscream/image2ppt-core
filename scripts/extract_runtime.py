"""Recover the supplied Node package bundle into this project only."""
import json
import tarfile
from pathlib import Path

archive = Path(r"C:\Users\83895\.cache\codex-runtimes\codex-runtime-install-PdivyH\node-runtime.tar.xz")
destination = Path(__file__).resolve().parents[1] / ".runtime" / "bundle"
destination.mkdir(parents=True, exist_ok=True)
prefix = "codex-primary-runtime/dependencies/node/node_modules/"
count = 0
with tarfile.open(archive) as bundle:
    for entry in bundle:
        if (entry.name.startswith(prefix)
                and not entry.name.startswith(prefix + ".pnpm/")
                and not entry.name.startswith(prefix + ".bin/")
                and (entry.isfile() or entry.isdir())):
            bundle.extract(entry, destination, filter="data")
            count += 1
print(json.dumps({"extracted_entries": count, "source": str(archive), "destination": str(destination)}))
