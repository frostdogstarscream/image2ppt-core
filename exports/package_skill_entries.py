"""Package each skill flavor from its own maintained source."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
variants = {
    "codex": (ROOT / "skills/image2ppt", ["SKILL.md", "references/slideir.md", "agents/openai.yaml"]),
    "claude": (ROOT / "skills/image2ppt-claude", ["SKILL.md", "references/slideir.md"]),
}


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    results = []
    for name, (source, files) in variants.items():
        archive = ROOT / "exports" / f"image2ppt-skill-{name}-{stamp}.zip"
        payload = {}
        for relative in files:
            item = source / relative
            item.resolve(strict=True).relative_to(ROOT.resolve())
            payload["image2ppt/" + relative] = item.read_bytes()
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
            for relative, content in payload.items():
                bundle.writestr(relative, content)
        with ZipFile(archive) as bundle:
            assert bundle.testzip() is None, "ZIP CRC check failed"
            assert set(bundle.namelist()) == set(payload), "Unexpected archive files"
            for relative, content in payload.items():
                assert bundle.read(relative) == content, "Packaged content changed"
        results.append({"variant": name, "archive": str(archive), "files": list(payload),
                        "bytes": archive.stat().st_size,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        "verified": True})
    report = ROOT / "exports" / f"image2ppt-skill-packaging-{stamp}.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packages": results, "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
