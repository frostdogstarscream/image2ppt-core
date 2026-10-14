import shutil
from pathlib import Path
from .storage import digest, inventory, local_file, read_json, write_json


def import_bundle(file, destination):
    """Normalize actual files, never accept model-supplied computed hashes."""
    file, destination = Path(file), Path(destination)
    ir = read_json(file)
    source = local_file(file.parent, ir['source_image'])
    if digest(source) != ir['source_sha256']:
        raise ValueError("source hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source").mkdir()
    (destination / "assets").mkdir()
    shutil.copyfile(source, destination / "source/page.png")
    ir['source_image'] = 'source/page.png'
    for asset in ir['assets']:
        original = local_file(file.parent, asset['path'])
        if digest(original) != asset['sha256']:
            raise ValueError("asset hash mismatch: " + asset['id'])
        # IDs have already passed the common JS validator.
        relative = 'assets/' + asset['id'] + original.suffix.lower()
        shutil.copyfile(original, destination / relative)
        asset['path'] = relative
        asset.pop('_file', None)
    write_json(destination / "slideir.json", ir, exclusive=True)
    files = ["slideir.json", "source/page.png"] + [a['path'] for a in ir['assets']]
    return inventory(destination, files)
