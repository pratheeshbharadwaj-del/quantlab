from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import cloud_storage

ENV = {"QNT_S3_ENDPOINT": "https://x.r2.cloudflarestorage.com", "QNT_S3_BUCKET": "b", "QNT_S3_ACCESS_KEY_ID": "id", "QNT_S3_SECRET_ACCESS_KEY": "secret"}


class CloudStorageTests(unittest.TestCase):
    def test_disabled_without_env(self) -> None:
        with patch.dict("os.environ", {k: "" for k in ENV}):
            self.assertFalse(cloud_storage.is_enabled())
            response = TestClient(app).post("/models/presign", json={"filename": "m.gguf"})
            self.assertEqual(response.status_code, 501)
            self.assertFalse(TestClient(app).get("/health").json()["cloud_storage"])

    def test_key_validation(self) -> None:
        self.assertTrue(cloud_storage.new_object_key("../../a b.GGUF").endswith("_a_b.gguf"))
        for bad in ("other/x.gguf", "models/../x", "models\\x"):
            with self.assertRaises(ValueError):
                cloud_storage.validate_key(bad)

    def test_presign_returns_url_and_rejects_bad_extension(self) -> None:
        with patch.dict("os.environ", ENV):
            self.assertTrue(TestClient(app).get("/health").json()["cloud_storage"])
            bad = TestClient(app).post("/models/presign", json={"filename": "evil.exe"})
            self.assertEqual(bad.status_code, 400)
            with patch.object(cloud_storage, "presign_upload", return_value="https://signed"):
                ok = TestClient(app).post("/models/presign", json={"filename": "m.gguf"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["upload_url"], "https://signed")
            self.assertTrue(ok.json()["key"].startswith("models/"))

    def test_register_downloads_and_imports(self) -> None:
        def fake_download(key: str, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"gguf-fixture")

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", ENV):
            with patch("app.main.MODEL_ROOT", Path(tmp).resolve()), patch.object(cloud_storage, "object_size", return_value=12), patch.object(cloud_storage, "download_to", side_effect=fake_download), patch("app.main.manager.import_model", return_value={"id": "1", "supported": False}) as imported:
                response = TestClient(app).post("/models/register", json={"key": "models/abc_m.gguf", "filename": "m.gguf", "role": "original"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["storage_key"], "models/abc_m.gguf")
            imported.assert_called_once()

    def test_register_rejects_foreign_key(self) -> None:
        with patch.dict("os.environ", ENV):
            response = TestClient(app).post("/models/register", json={"key": "secrets/x", "filename": "m.gguf"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
