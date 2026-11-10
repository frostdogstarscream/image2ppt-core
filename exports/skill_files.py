"""Shared, bounded skill payload inventory for install and export."""
from pathlib import Path


def skill_files(source):
    source=Path(source).resolve(strict=True)
    files=[source/'SKILL.md']
    for folder in ('references','agents'):
        files.extend(p for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.md','.yaml','.yml'))
    for file in files:
        file.resolve(strict=True).relative_to(source)
    return sorted(p.relative_to(source).as_posix() for p in files)
