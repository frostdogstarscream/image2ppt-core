"""Read the actual Office shapes and probe six manually identified editing roles."""

from pathlib import Path
import argparse, json, shutil
from PIL import Image
import numpy as np
import win32com.client


def normalized(s):
    return "".join(s.split())


def box(sh, sw, shh):
    return [
        sh.Left * 1672 / sw,
        sh.Top * 941 / shh,
        sh.Width * 1672 / sw,
        sh.Height * 941 / shh,
    ]


def intersects(a, b):
    x, y, w, h = a
    u, v, p, q = b
    return max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))


def name_text(s):
    try:
        return (
            s.TextFrame.TextRange.Text if s.HasTextFrame and s.TextFrame.HasText else ""
        )
    except Exception:
        return ""


def editable_shapes(shapes, depth=0):
    for shape in shapes:
        yield shape
        if shape.HasTable:
            table = shape.Table
            for row in range(1, table.Rows.Count + 1):
                for col in range(1, table.Columns.Count + 1):
                    yield table.Cell(row, col).Shape
        if shape.Type == 6 and depth < 5:
            yield from editable_shapes(shape.GroupItems, depth + 1)


def run(pptx, out, cases, single_page=None):
    out.mkdir(exist_ok=True, parents=True)
    app = win32com.client.DispatchEx("PowerPoint.Application")
    results = []
    try:
        for idx, (pg, label, target, action, value) in enumerate(cases):
            if single_page is not None and pg != single_page:
                continue
            if action not in ("text", "move", "delete", "fill"):
                raise ValueError(f"Unknown Office edit: {action}")
            f = out / f"{idx}-edit.pptx"
            single = pptx.is_dir()
            source = next(pptx.glob(f"{pg:02d}-*.pptx")) if single else pptx
            shutil.copyfile(source, f)
            pres = app.Presentations.Open(str(f.resolve()), False, False, False)
            try:
                slide = pres.Slides(1 if single or single_page is not None else pg)
                sw = pres.PageSetup.SlideWidth
                shh = pres.PageSetup.SlideHeight
                matches = []
                for s in editable_shapes(slide.Shapes):
                    b = box(s, sw, shh)
                    over = intersects(b, target)
                    area = b[2] * b[3]
                    ta = target[2] * target[3]
                    if min(area, ta) <= 0 or over / min(area, ta) < 0.55:
                        continue
                    if action == "move" and (
                        s.Type not in (1, 6) or over / max(1, area + ta - over) < 0.6
                    ):
                        continue
                    if action == "text" and normalized(name_text(s)) != normalized(
                        label
                    ):
                        continue
                    if action == "delete" and (
                        s.Type != 13 or over / max(1, area + ta - over) < 0.58
                    ):
                        continue
                    if action == "fill" and (
                        s.Type != 1 or over / max(1, area + ta - over) < 0.65
                    ):
                        continue
                    matches.append((over / (area + ta - over), s, b))
                row = {"page": pg, "target": label, "action": action}
                results.append(row)
                if not matches:
                    row["status"] = "NO_TARGET_OBJECT"
                    continue
                _, shape, b = max(matches, key=lambda item: item[0])
                row["object"] = shape.Name
                before = out / f"{idx}-before.png"
                after = out / f"{idx}-after.png"
                slide.Export(str(before.resolve()), "PNG", 1672, 941)
                if action == "text":
                    shape.TextFrame.TextRange.Text = value
                elif action == "move":
                    labels = [
                        s
                        for s in editable_shapes(slide.Shapes)
                        if normalized(name_text(s)) == normalized(label)
                        and intersects(box(s, sw, shh), target) > 0
                    ]
                    if not labels:
                        row["status"] = "NO_TARGET_LABEL"
                        continue
                    linked = normalized(name_text(shape)) == normalized(label) or (
                        shape.Type == 6
                        and any(
                            normalized(name_text(s)) == normalized(label)
                            for s in editable_shapes(shape.GroupItems)
                        )
                    )
                    mover = (
                        shape
                        if linked
                        else slide.Shapes.Range(
                            tuple(
                                dict.fromkeys([shape.Name] + [s.Name for s in labels])
                            )
                        )
                    )
                    row["selection"] = "one object" if linked else "multiple selection"
                    row["label_x_before"] = min(box(s, sw, shh)[0] for s in labels)
                    row["card_x_before"] = b[0]
                    mover.IncrementLeft(8 * sw / 1672)
                elif action == "delete":
                    shape.Delete()
                else:
                    shape.Fill.ForeColor.RGB = 0xEAD5B5
                pres.Save()
                pres.Close()
                pres = None
                pres = app.Presentations.Open(str(f.resolve()), True, False, False)
                slide = pres.Slides(1 if single or single_page is not None else pg)
                slide.Export(str(after.resolve()), "PNG", 1672, 941)
                if action == "text":
                    row["saved_value"] = any(
                        normalized(name_text(s)) == normalized(value)
                        and intersects(box(s, sw, shh), b) > 0
                        for s in editable_shapes(slide.Shapes)
                    )
                if action == "move":
                    shifted = [
                        s
                        for s in editable_shapes(slide.Shapes)
                        if normalized(name_text(s)) == normalized(label)
                        and intersects(box(s, sw, shh), target) > 0
                    ]
                    row["saved_value"] = bool(
                        shifted
                        and abs(
                            min(box(s, sw, shh)[0] for s in shifted)
                            - row["label_x_before"]
                            - 8
                        )
                        < 1
                        and abs(
                            box(slide.Shapes(row["object"]), sw, shh)[0]
                            - row["card_x_before"]
                            - 8
                        )
                        < 1
                    )
                if action == "delete":
                    row["saved_value"] = all(
                        s.Name != row["object"] for s in editable_shapes(slide.Shapes)
                    )
                if action == "fill":
                    row["saved_value"] = (
                        slide.Shapes(row["object"]).Fill.ForeColor.RGB == 0xEAD5B5
                    )
                a = np.asarray(Image.open(before).convert("RGB"), dtype=int)
                d = (
                    np.abs(
                        a - np.asarray(Image.open(after).convert("RGB"), dtype=int)
                    ).max(axis=2)
                    > 8
                )
                changed = int(d.sum())
                x, y, w, h = b
                d[
                    max(0, int(y) - 4) : int(y + h) + 5,
                    max(0, int(x) - 4) : int(x + w) + 15,
                ] = False
                row.update(
                    changed_pixels=changed,
                    outside_pixels=int(d.sum()),
                    status=(
                        "EDIT_OBSERVED"
                        if changed and not d.any() and row.get("saved_value", True)
                        else "REVIEW_REQUIRED"
                    ),
                )
            finally:
                if pres is not None:
                    pres.Close()
                if results[-1].get("status") == "NO_TARGET_OBJECT":
                    f.unlink()
    finally:
        if app.Presentations.Count == 0:
            app.Quit()
    (out / "report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print([(r["page"], r["status"]) for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pptx", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--single-page", type=int, help="Page number of a standalone one-slide deck"
    )
    args = parser.parse_args()
    run(
        args.pptx,
        args.out,
        json.loads(args.cases.read_text(encoding="utf-8-sig")),
        args.single_page,
    )
