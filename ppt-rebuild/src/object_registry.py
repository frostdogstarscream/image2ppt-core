"""Source evidence ownership and the per-object replacement lifecycle."""
from __future__ import annotations

import math


STATES = {
    "candidate": {"matched", "reverted"},
    "matched": {"trial_rendered", "reverted"},
    "trial_rendered": {"verified", "reverted"},
    "verified": {"committed", "reverted"},
    "committed": set(),
    "reverted": set(),
}


def validate_bbox(bbox, size):
    if len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
        raise ValueError("bbox must contain four finite source-pixel values")
    if any(int(v) != v for v in bbox):
        raise ValueError("mask/crop bbox boundaries must be integer source pixels")
    x, y, w, h = bbox
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > size[0] or y + h > size[1]:
        raise ValueError(f"bbox outside source canvas: {bbox}")


class ObjectRegistry:
    def __init__(self, size, source_sha256):
        self.size = list(size)
        self.source_sha256 = source_sha256
        self.objects = []
        self.operations = []

    def add(self, candidate):
        validate_bbox(candidate["bbox"], self.size)
        if any(o["id"] == candidate["id"] for o in self.objects):
            raise ValueError(f"duplicate object id: {candidate['id']}")
        entry = dict(candidate, state="candidate", final_owner="background-image")
        entry["evidence_id"] = f"ev-{entry['id']}"
        self.objects.append(entry)
        return entry

    def transition(self, entry, state, **details):
        previous = entry["state"]
        if state not in STATES[previous]:
            raise ValueError(f"invalid transition {previous} -> {state}")
        self.operations.append({"operation_id": len(self.operations) + 1,
                                "object_id": entry["id"], "from": previous,
                                "to": state, "source_sha256": self.source_sha256, **details})
        entry.update(state=state, **details)
        if state == "reverted":
            entry["final_owner"] = "background-image"
        elif state == "committed":
            entry["final_owner"] = entry["id"]

    def validate_owners(self, ir):
        owners = {}
        assets = {a["id"]: a for a in ir["assets"]}
        claims = [(n["id"], n["evidence_ids"]) for n in ir["nodes"]]
        for node in ir["nodes"]:
            if node["type"] == "image":
                claims.append((node["id"], assets[node["asset_id"]]["contains_evidence_ids"]))
        if ir["background"].get("asset_id"):
            claims.append(("background-image", assets[ir["background"]["asset_id"]]["contains_evidence_ids"]))
        for owner, ids in claims:
            for evidence_id in ids:
                if evidence_id in owners:
                    raise ValueError(f"duplicate output owner for {evidence_id}")
                owners[evidence_id] = owner
        if set(owners) != {e["id"] for e in ir["evidence"]}:
            raise ValueError("missing or unknown source evidence ownership")
        for entry in self.objects:
            expected = entry["id"] if entry["state"] in {"matched", "trial_rendered", "verified", "committed"} else "background-image"
            if owners[entry["evidence_id"]] != expected:
                raise ValueError(f"wrong owner for {entry['id']}")
        return owners

    def record(self):
        return {"schema": "ObjectRegistry/1", "source_sha256": self.source_sha256,
                "canvas_px": self.size, "coordinate_space": "source_pixels_xywh",
                "objects": self.objects, "operations": self.operations}
