# Page-level typography refinement

Use the most recently delivered scenes and the original source pages. The
conversation authors roles from the source; this path never calls OCR or a model
provider. `page_typography.py` accepts an internal page map, not a new SlideIR.

Each page has groups with `id`, paragraph `members` (`node-id/p0`), source-reviewed
`families`, `bold`, and nominal `size_pt`. Optional `exceptions` bind explicit run
emphasis (`node-id/p0/r1`). Every paragraph belongs to exactly one group. Omitted
run overrides inherit the group; content, color, frames, groups, source assets and
all non-text nodes remain unchanged. Font files are discovered from installed
Windows fonts, using real weights, character coverage and Windows family names.

```powershell
python ppt-rebuild/src/page_typography.py prepare --baseline <delivery-root> --plan <page-map.json> --output <new-run>
python ppt-rebuild/src/page_typography.py compose --output <new-run>
```

`prepare` ranks shared candidates against source glyphs, then applies capacity
filters to whole groups. A failed candidate is never partially applied; all
candidates failing retains the entire old group. Explicit emphasis must remain
available in the installed font manifest. `compose` uses the fixed compiler and
verifies all effective paragraph/run styles in the final package. Compilation is
an intermediate build, not visual acceptance or publication.

Before delivery, open the actual deck in PowerPoint, export every page, inspect
the source and full-page style, and check short edits/save/reopen on a copy. Use
`verify_rendered_change` with same-renderer old/new previews and independently
text-deleted decks to reject changed non-text content or changed pixels outside
the text regions. Review in-frame neighboring objects and all text completeness
visually; containment alone does not prove collision freedom. A failed actual-file
check requires trying the next candidate for the affected whole group, or keeping
its baseline and reporting the limitation. Repeat final-deck checks after changes.

Deliver only the new PPTX and actual PowerPoint page previews. Keep intermediate
build data outside the delivery directory. Do not add review labels to slides or
claim screenshot-based font selection recovered original font metadata.

Validation: `python -m pytest ppt-rebuild/tests/test_typography_groups.py
ppt-rebuild/tests/test_typography.py -q` and `./image2ppt.ps1 test`.
