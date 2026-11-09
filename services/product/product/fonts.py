# Font discovery shared with the verified page-font-fidelity implementation.
from pathlib import Path
from PIL import ImageFont
from hashlib import sha256
def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def windows_family_name(names):
    """Use the family name resolved by Windows/Office, not a Mac alias."""
    record=names.getName(1,3,1,0x409)
    if record is None:
        record=next((n for n in names.names if n.nameID==1 and n.platformID==3),None)
    return record.toUnicode() if record is not None else names.getDebugName(1)


def font_manifest(texts, directory=None):
    """Discover actual static faces, including TTC indices and real weights.

    Keep the FontManifest/1 contract. Never synthesize a missing bold face.
    Private font files not installed into Windows are deliberately excluded.
    """
    import os
    from fontTools.ttLib import TTFont, TTCollection, TTLibError
    directories = [Path(directory)] if directory else [Path('C:/Windows/Fonts'),
        Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'Microsoft/Windows/Fonts']
    chars = set(''.join(texts)) - set('\n\r\t')
    records = []
    aliases = {'Microsoft YaHei':'微软雅黑','Microsoft YaHei UI':'微软雅黑 UI',
               'DengXian':'等线','SimHei':'黑体','SimSun':'宋体'}
    for file in sorted({p for d in directories if d.exists() for p in d.iterdir()
                        if p.suffix.lower() in {'.ttf','.otf','.ttc'}}):
        try:
            count = 1
            if file.suffix.lower() == '.ttc':
                collection = TTCollection(file, lazy=True)
                count = len(collection.fonts); collection.close()
            file_hash = None
            for index in range(count):
                with TTFont(file, fontNumber=index, lazy=True) as face:
                    if 'fvar' in face or 'OS/2' not in face: continue
                    os2 = face['OS/2']; weight = os2.usWeightClass
                    if os2.fsSelection & 1 or weight not in (400,700): continue
                    # PowerPoint resolves Windows legacy family names, not the
                    # shorter Mac/typographic family (e.g. Lenovo's 60S face).
                    family = windows_family_name(face['name'])
                    if not family: continue
                    family = aliases.get(family, family)
                    cmap = face.getBestCmap() or {}
                    font = ImageFont.truetype(str(file), 32, index=index)
                    font.getmask('Aa09中文')
                    file_hash = file_hash or digest(file)
                    records.append({'family':family,'file':str(file),'index':index,'bold':weight==700,
                        'real_name':list(font.getname()),'sha256':file_hash,
                        'weight':weight,'cjk':all(ord(c) in cmap for c in '中文业务'),
                        'sans':getattr(os2.panose,'bSerifStyle',0) in (0,1,11,12,13),
                        'missing_characters':''.join(sorted(c for c in chars if ord(c) not in cmap))})
        except (OSError, ValueError, KeyError, TTLibError):
            continue
    # Match family/weight the same way PowerPoint does, without duplicate aliases.
    records = list({(r['family'],r['bold']):r for r in reversed(records)}.values())
    if not records: raise ValueError('No verified fonts')
    return {'schema':'FontManifest/1','fonts':records}
