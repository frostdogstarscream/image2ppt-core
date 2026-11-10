import importlib.util
import io
import json
import sys
import shutil
import uuid
import types
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

spec = importlib.util.spec_from_file_location("layerd_service", Path(__file__).with_name("app.py"))
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class ServiceTests(unittest.TestCase):
    def test_configured_iteration_cap_preserves_default_and_early_stop(self):
        with patch.dict(os.environ, {"LAYERD_MAX_ITERATIONS": "12"}):
            custom_spec = importlib.util.spec_from_file_location("layerd_service_4090", Path(__file__).with_name("app.py"))
            custom = importlib.util.module_from_spec(custom_spec)
            custom_spec.loader.exec_module(custom)
        self.assertEqual(custom.capabilities()["max_iterations"], [1, 12])
        schema = custom.app.openapi()
        body = schema["components"]["schemas"]["Body_decompose_v1_decompose_post"]["properties"]["max_iterations"]
        self.assertEqual(body["maximum"], 12)
        self.assertEqual(body["default"], 3)
        import numpy as np
        image = Image.new("RGB", (20, 10), "white")
        model = types.SimpleNamespace(matting_model=lambda image: np.zeros((10, 20)),
            fg_refine=True, bg_refine=True, _decompose_step=lambda image: (None, image))
        layers, steps = custom.bounded_decompose(model, image, 12)
        self.assertEqual(len(layers), 1)
        self.assertEqual(len(steps), 1)

    def setUp(self):
        self.test_base = Path(__file__).resolve().parents[2] / ".tmp-test" / "layerd-service"
        self.root = self.test_base / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        (self.root / "jobs").mkdir()
        (self.root / "token").write_text("t" * 64)
        self.patches = [patch.object(service, "ROOT", self.root),
                        patch.object(service, "TOKEN_FILE", self.root / "token"),
                        patch.object(service, "THREAD", types.SimpleNamespace(is_alive=lambda: True))]
        for p in self.patches:
            p.start()
        service.STATE.update(ready=True, loading=False, error_type=None, current_job=None)
        self.client = TestClient(service.app)
        self.auth = {"Authorization": "Bearer " + "t" * 64}

    def tearDown(self):
        while not service.JOBS.empty():
            service.JOBS.get_nowait()
            service.JOBS.task_done()
        for p in reversed(self.patches):
            p.stop()
        self.root.resolve().relative_to(self.test_base.resolve())
        shutil.rmtree(self.root)

    def test_auth_and_not_ready(self):
        self.assertEqual(self.client.get("/v1/capabilities").status_code, 401)
        service.STATE["ready"] = False
        self.assertEqual(self.client.get("/ready").status_code, 503)
        result = self.client.post("/v1/decompose", headers=self.auth, files={"file": ("x.png", b"bad")})
        self.assertEqual(result.status_code, 503)
        self.assertEqual(list((self.root / "jobs").iterdir()), [])

    def test_bound_source_is_enqueued_once_even_after_client_retry(self):
        with patch.dict(service.STATE,{'model_binding':'synthetic-model-sha'}):
            image=io.BytesIO();Image.new('RGB',(20,10),'white').save(image,format='PNG')
            ids=[]
            for _ in range(2):
                r=self.client.post('/v1/decompose',headers=self.auth,files={'file':('x.png',image.getvalue())})
                self.assertEqual(r.status_code,202);ids.append(r.json()['id'])
            self.assertEqual(ids[0],ids[1]);self.assertEqual(service.JOBS.qsize(),1)

    def test_upload_contract_and_artifact_allowlist(self):
        result = self.client.post("/v1/decompose", headers=self.auth, files={"file": ("x.png", b"bad")})
        self.assertEqual(result.status_code, 400)
        image = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(image, format="PNG")
        result = self.client.post("/v1/decompose", headers=self.auth,
                                  files={"file": ("x.png", image.getvalue())}, data={"max_iterations": 6})
        self.assertEqual(result.status_code, 422)
        result = self.client.post("/v1/decompose", headers=self.auth,
                                  files={"file": ("x.png", image.getvalue())}, data={"max_iterations": 2})
        self.assertEqual(result.status_code, 202)
        job_id = result.json()["id"]
        self.assertEqual(self.client.get(f"/v1/jobs/{job_id}", headers=self.auth).json()["status"], "QUEUED")
        self.assertEqual(self.client.get(f"/v1/jobs/{job_id}/artifacts/original.bin", headers=self.auth).status_code, 404)
        self.assertEqual(self.client.get("/v1/jobs/not-a-job", headers=self.auth).status_code, 404)

    def test_saved_layer_order_and_composite(self):
        job_id = "a" * 32
        path = self.root / "jobs" / job_id
        path.mkdir()
        bottom = Image.new("RGBA", (20, 10), "blue")
        top = Image.new("RGBA", (20, 10), (255, 0, 0, 128))
        expected = Image.alpha_composite(bottom, top)
        expected.convert("RGB").save(path / "source.png")
        service.write_json(path / "request.json", {"max_iterations": 2, "source_sha256": "example"})
        import numpy as np
        calls = []
        def step(image):
            calls.append(1)
            return (top, bottom.convert("RGB")) if len(calls) == 1 else (None, image)
        model = types.SimpleNamespace(matting_model=lambda image: np.zeros((10, 20)),
                                      fg_refine=True, bg_refine=True, _decompose_step=step)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0),
            inference_mode=lambda: __import__("contextlib").nullcontext())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            service.process(model, path)
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest["layer_order"], "back_to_front")
        self.assertEqual(manifest["composite_mae_255"], 0)
        self.assertEqual(manifest["layers"][1]["partial_alpha_fraction"], 1)
        self.assertFalse(manifest["native_text"])
        self.assertEqual(json.loads((path / "status.json").read_text())["status"], "SUCCEEDED")

    def test_dense_components_skip_optional_refinement_and_restore_model(self):
        import numpy as np
        alpha = np.zeros((100, 100), dtype=float)
        alpha[::3, ::3] = 1
        observed = []
        image = Image.new("RGB", (100, 100), "white")
        model = types.SimpleNamespace(matting_model=lambda image: alpha, fg_refine=True, bg_refine=True)
        original = model.matting_model
        def step(current):
            observed.append((model.fg_refine, model.bg_refine))
            return image.convert("RGBA"), current
        model._decompose_step = step
        layers, steps = service.bounded_decompose(model, image, 1)
        self.assertEqual(observed, [(False, False)])
        self.assertEqual(steps[0]["reason"], "component_memory_budget")
        self.assertIs(model.matting_model, original)
        self.assertTrue(model.fg_refine and model.bg_refine)

    def test_nested_color_components_retry_without_refinement(self):
        import numpy as np
        dense = np.zeros((100, 100), dtype=bool)
        dense[::3, ::3] = True
        image = Image.new("RGB", (100, 100), "white")
        observed = []
        model = types.SimpleNamespace(matting_model=lambda image: np.ones((100, 100)),
                                      fg_refine=True, bg_refine=True)
        def step(current):
            observed.append((model.fg_refine, model.bg_refine))
            if model.fg_refine:
                service.budgeted_components(dense)
            return image.convert("RGBA"), current
        model._decompose_step = step
        _, steps = service.bounded_decompose(model, image, 1)
        self.assertEqual(observed, [(True, True), (False, False)])
        self.assertEqual(steps[0]["reason"], "nested_component_memory_budget")
        self.assertGreater(steps[0]["nested_budget"]["components"], 128)
        self.assertTrue(model.fg_refine and model.bg_refine)


if __name__ == "__main__":
    unittest.main()
