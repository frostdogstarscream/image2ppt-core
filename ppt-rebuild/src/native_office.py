"""Read native text bounds from a saved PowerPoint presentation."""

import json
import sys
import os
import shutil
import subprocess
from pathlib import Path


def measure_office_fields(pptx):
    try:
        import win32com.client
    except ImportError:
        interpreters = [
            os.getenv("IMAGE2PPT_OFFICE_PYTHON"),
            shutil.which("python"),
            getattr(sys, "_base_executable", None),
        ]
        if __name__ != "__main__" and os.name == "nt":
            for executable in dict.fromkeys(interpreters):
                if not executable or Path(executable).resolve() == Path(sys.executable).resolve():
                    continue
                try:
                    child = subprocess.run(
                        [executable, "-X", "utf8", __file__, str(pptx)],
                        capture_output=True,
                        timeout=100,
                        check=True,
                    )
                    result = json.loads(child.stdout)
                    if result["status"] == "measured":
                        return result
                except (OSError, ValueError, subprocess.SubprocessError):
                    continue
        return {"status": "unavailable", "reason": "Office COM interpreter unavailable", "fields": []}
    app = deck = None
    measurements = []
    try:
        app = win32com.client.DispatchEx("PowerPoint.Application")
        deck = app.Presentations.Open(str(pptx), ReadOnly=True, WithWindow=False)

        def inspect(shapes):
            for shape in shapes:
                if shape.Type == 6:
                    inspect(shape.GroupItems)
                elif shape.HasTextFrame and shape.TextFrame.HasText and shape.Name.startswith("text-"):
                    r = shape.TextFrame.TextRange
                    measurements.append(
                        dict(
                            id=shape.Name,
                            width=shape.Width,
                            height=shape.Height,
                            ink_width=r.BoundWidth,
                            ink_height=r.BoundHeight,
                            line_count=r.Lines().Count,
                        )
                    )

        inspect(deck.Slides(1).Shapes)
        return {"status": "measured", "fields": measurements}
    except Exception as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "fields": []}
    finally:
        if deck is not None:
            deck.Close()
        if app is not None and app.Presentations.Count == 0:
            app.Quit()


if __name__ == "__main__":
    print(json.dumps(measure_office_fields(sys.argv[1])))
