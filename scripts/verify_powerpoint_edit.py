"""Verify real PowerPoint edits on a copy of one compiled page.

The input PPTX is never saved. A separate test copy is moved, edited, saved,
reopened, and inspected through PowerPoint COM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import win32com.client


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shapes(slide) -> dict:
    return {slide.Shapes.Item(i).Name: slide.Shapes.Item(i)
            for i in range(1, slide.Shapes.Count + 1)}


def verify(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an edit check")
    output.mkdir(parents=True)
    copy = output / "page02-edit-check.pptx"
    shutil.copy2(source, copy)
    input_hash = sha256(source)
    app = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = None
    try:
        presentation = app.Presentations.Open(str(copy), ReadOnly=False,
                                              Untitled=False, WithWindow=False)
        slide = presentation.Slides.Item(1)
        initial = shapes(slide)
        required = (["left-panel", "bottom-banner"] +
                    [f"right-panel-{i:02d}" for i in range(1, 6)] +
                    [f"checkbox-{i:02d}" for i in range(1, 4)] +
                    ["node-logo"] +
                    [f"node-icon-row{i}" for i in range(1, 6)] +
                    ["node-icon-compute", "node-icon-collab", "node-icon-app"])
        missing = sorted(set(required) - set(initial))
        if missing:
            raise AssertionError(f"independent shapes missing: {missing}")
        before = {name: (initial[name].Left, initial[name].Top,
                         initial[name].Width, initial[name].Height)
                  for name in required}
        initial["left-panel"].Left += 12
        initial["right-panel-03"].Top += 8
        initial["checkbox-02"].Delete()
        text = initial["t-row2-desc"].TextFrame.TextRange
        old_text = text.Text
        text.Text = old_text + "【编辑验证】"
        presentation.Save()
        presentation.Close()
        presentation = app.Presentations.Open(str(copy), ReadOnly=True,
                                              Untitled=False, WithWindow=False)
        after = shapes(presentation.Slides.Item(1))
        assert "checkbox-02" not in after
        assert abs(after["left-panel"].Left - before["left-panel"][0] - 12) < .1
        assert abs(after["right-panel-03"].Top - before["right-panel-03"][1] - 8) < .1
        assert after["t-row2-desc"].TextFrame.TextRange.Text == old_text + "【编辑验证】"
        for name in ("right-panel-01", "right-panel-02", "node-logo", "node-icon-app"):
            actual = (after[name].Left, after[name].Top, after[name].Width, after[name].Height)
            assert all(abs(a - b) < .1 for a, b in zip(actual, before[name])), name
        assert sha256(source) == input_hash, "original PPTX changed"
        result = {"status": "PASS", "application": "Microsoft PowerPoint COM",
                  "source_pptx": str(source), "source_sha256": input_hash,
                  "edited_copy": str(copy), "edited_copy_sha256": sha256(copy),
                  "independent_shapes_checked": required,
                  "operations": ["left panel moved", "third right panel moved",
                                 "second checkbox deleted", "description text edited",
                                 "saved and reopened", "four sibling positions unchanged"],
                  "human_visual_acceptance": False, "wps_checked": False}
        (output / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if presentation is not None:
            presentation.Close()
        app.Quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.pptx, args.output), ensure_ascii=False, indent=2))
