from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.evaluation.metrics import exact_match, output_similarity, score_outputs, token_f1
from app.main import app
from app.services.model_detector import detect_model
from app.services.model_manager import ModelManager


class MetricTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(exact_match("A model!", "a model"), 1.0)
        self.assertEqual(exact_match("a model", "another model"), 0.0)

    def test_token_f1_is_overlap_based(self) -> None:
        self.assertAlmostEqual(token_f1("a quantized model", "a model"), 0.8)
        self.assertEqual(token_f1("left", "right"), 0.0)

    def test_output_similarity_is_lexical_jaccard(self) -> None:
        self.assertAlmostEqual(output_similarity("red blue", "blue green") or 0, 1 / 3)
        self.assertEqual(output_similarity("", ""), 1.0)

    def test_missing_reference_scores_are_unavailable(self) -> None:
        self.assertEqual(score_outputs([{"output": "answer"}]), {"exact_match": None, "token_f1": None})


class DetectorTests(unittest.TestCase):
    def test_detects_gguf_from_extension_without_inventing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"fixture")
            detected = detect_model(model)
            self.assertEqual(detected["format"], "GGUF")
            self.assertEqual(detected["size_bytes"], 7)
            self.assertIsNone(detected["quantization"])
            self.assertIsNone(detected["architecture"])

    def test_detects_transformers_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"architectures":["LlamaForCausalLM"],"model_type":"llama"}', encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            found = detect_model(root)
            self.assertEqual(found["format"], "SAFETENSORS")
            self.assertEqual(found["architecture"], "LlamaForCausalLM")
            self.assertFalse(found["chat_template_available"])
            self.assertEqual(found["size_bytes"], (root / "config.json").stat().st_size + (root / "model.safetensors").stat().st_size)

    def test_detects_chat_template_when_declared_by_local_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"architectures":["ExampleForCausalLM"],"model_type":"example"}', encoding="utf-8")
            (root / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            found = detect_model(root)
            self.assertTrue(found["chat_template_available"])

    def test_gpt2_chat_formatter_uses_current_question_not_old_turns(self) -> None:
        from app.runtimes.transformers_runtime import TransformersRuntime
        runtime = object.__new__(TransformersRuntime)
        runtime.model = type("Model", (), {"config": type("Config", (), {"model_type": "gpt2"})()})()
        runtime.tokenizer = type("Tokenizer", (), {"chat_template": None})()
        formatted = runtime.format_messages([
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old completion"},
            {"role": "user", "content": "What is a token?"},
        ])
        self.assertEqual(formatted, "Question: What is a token?\nAnswer:")

    def test_fingerprint_matches_identical_model_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = [root / "original", root / "quantized"]
            for directory in directories:
                directory.mkdir()
                (directory / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
                (directory / "weights.safetensors").write_bytes(b"same model weights")
            first, second = (detect_model(directory) for directory in directories)
            self.assertEqual(first["content_fingerprint"], second["content_fingerprint"])

    def test_model_paths_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            manager = ModelManager(root)
            with self.assertRaises(ValueError):
                manager.resolve_path("../outside.gguf")

    def test_import_extracts_local_hugging_face_zip_and_detects_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            archive_path = root / "local-model.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("config.json", '{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2","quantization_config":{"load_in_4bit":true,"bnb_4bit_quant_type":"nf4","quant_method":"bitsandbytes"}}')
                archive.writestr("model.safetensors", b"local weights")
                archive.writestr("tokenizer.json", "{}")
            imported = ModelManager(root).import_model("local-model.zip")
            self.assertEqual(imported["format"], "SAFETENSORS")
            self.assertEqual(imported["architecture"], "GPT2LMHeadModel")
            self.assertEqual(imported["quantization"], "4-bit NF4 (bitsandbytes)")
            self.assertEqual(imported["archive_source"], "local-model.zip")
            self.assertTrue((root / imported["path"] / "config.json").is_file())

    def test_zip_import_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "must not be written")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                ModelManager(root).import_model("unsafe.zip")
            self.assertFalse((Path(tmp) / "outside.txt").exists())


class ApiTests(unittest.TestCase):
    def test_builtin_evaluation_dataset_route_is_not_shadowed(self) -> None:
        response = TestClient(app).get("/evaluation/dataset")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()), 20)

    def test_health_reports_local_mode(self) -> None:
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "LOCAL MODE")

    def test_unloaded_model_generation_is_an_explicit_error(self) -> None:
        response = TestClient(app).post("/generate", json={"model_id": "missing", "prompt": "hello"})
        self.assertIn(response.status_code, {404, 422})

    def test_model_import_blocks_path_traversal(self) -> None:
        response = TestClient(app).post("/models/import", json={"path": "../outside.gguf"})
        self.assertEqual(response.status_code, 400)

    def test_model_import_validates_selected_format(self) -> None:
        fake_record = {"id": "test-id", "format": "GGUF"}
        with patch("app.main.manager.import_model", return_value=fake_record):
            response = TestClient(app).post("/models/import", json={"path": "test.gguf", "format_hint": "ONNX"})
        self.assertEqual(response.status_code, 400)

    def test_unknown_extension_returns_unsupported_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from app.main import manager
            previous_root = manager.model_root
            manager.model_root = Path(tmp).resolve()
            try:
                (Path(tmp) / "weights.unknown").write_bytes(b"local fixture")
                response = TestClient(app).post("/models/import", json={"path": "weights.unknown"})
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()["supported"])
                self.assertEqual(response.json()["compatibility_message"], "Unsupported model format for this device/runtime")
            finally:
                manager.model_root = previous_root

    def test_compare_rejects_the_same_model_for_both_roles(self) -> None:
        response = TestClient(app).post("/evaluation/compare", json={"original_model_id": "same", "quantized_model_id": "same", "prompt": "test"})
        self.assertEqual(response.status_code, 400)

    def test_compare_rejects_identical_weight_artifacts(self) -> None:
        from app.main import manager
        original = {"id": "original", "name": "original", "size_bytes": 10, "content_fingerprint": "same", "generation_supported": True}
        quantized = {"id": "quantized", "name": "quantized", "size_bytes": 5, "content_fingerprint": "same", "generation_supported": True}
        with patch("app.main.manager.get", side_effect=[original, quantized]), patch.dict(manager.loaded, {"original": object(), "quantized": object()}):
            response = TestClient(app).post("/evaluation/compare", json={"original_model_id": "original", "quantized_model_id": "quantized", "prompt": "test"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("identical model weights", response.json()["detail"])

    def test_compare_returns_real_outputs_and_metrics_for_distinct_loaded_models(self) -> None:
        from app.main import manager
        class Runtime:
            ram_increase_bytes = 100
            ram_before_load_bytes = 1000
            ram_after_load_bytes = 1100
            def perplexity(self, text: str) -> None:
                return None
        original = {"id": "original", "name": "Original", "size_bytes": 1000, "content_fingerprint": "weights-a", "generation_supported": True}
        quantized = {"id": "quantized", "name": "Quantized", "size_bytes": 500, "content_fingerprint": "weights-b", "generation_supported": True}
        def generate(model_id: str, prompt: str, *_args: object):
            text = "same answer" if model_id == "original" else "similar answer"
            return ({"text": text, "tokens_generated": 2, "tokens_per_second": 10.0, "generation_time": 0.2}, manager.loaded[model_id])
        with patch("app.main.manager.get", side_effect=[original, quantized, original, quantized]), patch.dict(manager.loaded, {"original": Runtime(), "quantized": Runtime()}), patch("app.main.manager.generate", side_effect=generate):
            response = TestClient(app).post("/evaluation/compare", json={"original_model_id": "original", "quantized_model_id": "quantized", "dataset": [{"id": 1, "question": "test question", "expected_answer": "same answer"}], "prompt": "same prompt"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["original"]["evaluation"]["metrics"]["exact_match"], 1.0)
        self.assertEqual(result["quantized"]["evaluation"]["metrics"]["exact_match"], 0.0)
        self.assertEqual(result["prompt_comparison"]["original_output"], "same answer")
        self.assertEqual(result["prompt_comparison"]["quantized_output"], "similar answer")

    def test_benchmark_returns_measured_result_without_token_limit_argument(self) -> None:
        from app.main import manager
        fake_runtime = object()
        generated = {"text": "actual model output", "tokens_generated": 3, "tokens_per_second": 30.0, "generation_time": 0.1}
        with patch.dict(manager.loaded, {"benchmark-model": fake_runtime}), patch("app.main.manager.generate", return_value=(generated, fake_runtime)):
            response = TestClient(app).post("/benchmark/run", json={"model_id": "benchmark-model", "prompt": "test", "runs": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["tokens_per_second_average"], 30.0)
        self.assertEqual(len(response.json()["runs"]), 1)

    def test_quantized_local_upload_attempts_auto_load(self) -> None:
        from app.main import manager
        with tempfile.TemporaryDirectory() as tmp:
            previous_root = manager.model_root
            manager.model_root = Path(tmp).resolve()
            record = {"id": "quant-id", "name": "local-quant", "supported": True, "loaded": False}
            try:
                with patch("app.main.manager.import_model", return_value=record), patch("app.main.manager.load", return_value={**record, "loaded": True}) as load:
                    response = TestClient(app).post("/models/upload", files={"file": ("weights.gguf", b"gguf-fixture", "application/octet-stream")}, data={"role": "quantized"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["role"], "quantized")
                self.assertTrue(response.json()["loaded"])
                load.assert_called_once_with("quant-id")
            finally:
                manager.model_root = previous_root

    def test_original_local_upload_does_not_auto_load(self) -> None:
        from app.main import manager
        with tempfile.TemporaryDirectory() as tmp:
            previous_root = manager.model_root
            manager.model_root = Path(tmp).resolve()
            record = {"id": "original-id", "name": "local-original", "supported": True, "loaded": False}
            try:
                with patch("app.main.manager.import_model", return_value=record), patch("app.main.manager.load") as load:
                    response = TestClient(app).post("/models/upload", files={"file": ("weights.gguf", b"gguf-fixture", "application/octet-stream")}, data={"role": "original"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["role"], "original")
                self.assertFalse(response.json()["loaded"])
                load.assert_not_called()
            finally:
                manager.model_root = previous_root


if __name__ == "__main__":
    unittest.main()
