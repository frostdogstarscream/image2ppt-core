from pathlib import Path
from .contracts import AuthoringRecord, BuildManifest, ProviderResult
from .storage import digest, local_file, read_json, verify_files

AUTHORING_FILE = "authoring.json"


class ManualDraftProvider:
    def load(self, file):
        file = Path(file).resolve(strict=True)
        ir = read_json(file)
        if ir.get("schema_version") != "0.1" or not all(k in ir for k in ("nodes", "assets", "source_image")):
            raise ValueError("M1 imports saved SlideIR 0.1 bundles, not raw model responses")
        return ProviderResult(kind="manual", ir_file=str(file), authoring=self.authoring(file))

    @staticmethod
    def authoring(file):
        """Load the authoring sidecar when present, and bind it to this IR.

        Absent means the IR predates the record (legacy runs stay replayable).
        Present but stale is a hard failure: the IR was edited after being
        recorded, so the recorded author describes something else.
        """
        record_path = file.parent / AUTHORING_FILE
        if not record_path.is_file():
            return None
        authoring = AuthoringRecord.model_validate(read_json(record_path))
        actual = digest(file)
        if authoring.ir_sha256 != actual:
            raise ValueError(
                f"{AUTHORING_FILE} does not describe this IR: recorded "
                f"{authoring.ir_sha256[:12]}, actual {actual[:12]}. Re-record the "
                f"authoring entry after finishing the IR."
            )
        return authoring


class ReplayProvider:
    def load(self, file):
        file = Path(file).resolve(strict=True)
        build = BuildManifest.model_validate(read_json(file))
        verify_files(file.parent, build.files)
        for required in ("slideir.json", "result.pptx", "preview.png", "review.json", "objects.json"):
            if required not in build.files:
                raise ValueError("build lacks required artifact: " + required)
        return ProviderResult(kind="replay", ir_file=str(local_file(file.parent, "slideir.json")), previous_environment=build.environment)
