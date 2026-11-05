"""Shared test helpers for ppt-rebuild."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # ppt-rebuild/
REPO = ROOT.parent                                   # image2PPT/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PAGES = REPO / "outputs" / "layerd-review-20261104"
CONFIG = ROOT / "configs" / "rebuild.yaml"


def page_dir(name: str) -> Path:
    matches = [p for p in PAGES.iterdir() if p.is_dir() and p.name.startswith(name)]
    if not matches:
        raise FileNotFoundError(f"no page starting with {name} under {PAGES}")
    return matches[0]


def load_config() -> dict:
    text = CONFIG.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is present in the bundled runtime
        raise RuntimeError("PyYAML is required to read configs/rebuild.yaml")
    return yaml.safe_load(text)


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
