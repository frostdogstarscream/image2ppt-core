from pathlib import Path
from .contracts import BuildManifest, ProviderResult
from .storage import local_file, read_json, verify_files


class ManualDraftProvider:
    def load(self, file):
        file = Path(file).resolve(strict=True)
        ir = read_json(file)
        if ir.get("schema_version") != "0.1" or not all(k in ir for k in ("nodes", "assets", "source_image")):
            raise ValueError("M1 imports saved SlideIR 0.1 bundles, not raw model responses")
        return ProviderResult(kind="manual", ir_file=str(file))


class ReplayProvider:
    def load(self, file):
        file = Path(file).resolve(strict=True)
        build = BuildManifest.model_validate(read_json(file))
        verify_files(file.parent, build.files)
        for required in ("slideir.json", "result.pptx", "preview.png", "review.json", "objects.json"):
            if required not in build.files:
                raise ValueError("build lacks required artifact: " + required)
        return ProviderResult(kind="replay", ir_file=str(local_file(file.parent, "slideir.json")), previous_environment=build.environment)
