"""S3-compatible object storage (Cloudflare R2, AWS S3, Backblaze, MinIO).

Enabled only when all QNT_S3_* variables are set. Credentials come from the
environment and are never hardcoded. The bucket stays private; browsers upload
with short-lived pre-signed PUT URLs.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

PRESIGN_SECONDS = 3600
KEY_PREFIX = "models/"


def _settings() -> dict[str, str] | None:
    names = {"endpoint": "QNT_S3_ENDPOINT", "bucket": "QNT_S3_BUCKET", "access_key": "QNT_S3_ACCESS_KEY_ID", "secret_key": "QNT_S3_SECRET_ACCESS_KEY"}
    values = {key: os.environ.get(env, "").strip() for key, env in names.items()}
    if not all(values.values()):
        return None
    values["region"] = os.environ.get("QNT_S3_REGION", "auto").strip() or "auto"
    return values


def is_enabled() -> bool:
    return _settings() is not None


def _client() -> tuple[Any, str]:
    settings = _settings()
    if settings is None:
        raise RuntimeError("Cloud storage is not configured. Set the QNT_S3_* environment variables (see .env.example).")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Run: pip install boto3") from exc
    client = boto3.client(
        "s3",
        endpoint_url=settings["endpoint"],
        aws_access_key_id=settings["access_key"],
        aws_secret_access_key=settings["secret_key"],
        region_name=settings["region"],
        config=Config(signature_version="s3v4"),
    )
    return client, settings["bucket"]


def new_object_key(filename: str) -> str:
    path = Path(filename.replace("\\", "/").rsplit("/", 1)[-1])
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "model"
    return f"{KEY_PREFIX}{uuid.uuid4().hex}_{safe}{path.suffix.lower()}"


def validate_key(key: str) -> str:
    if not key.startswith(KEY_PREFIX) or ".." in key or "\\" in key or len(key) > 300:
        raise ValueError("Invalid storage key.")
    return key


def presign_upload(key: str) -> str:
    client, bucket = _client()
    return client.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": validate_key(key)}, ExpiresIn=PRESIGN_SECONDS)


def object_size(key: str) -> int:
    client, bucket = _client()
    try:
        return int(client.head_object(Bucket=bucket, Key=validate_key(key))["ContentLength"])
    except Exception as exc:
        raise ValueError("Uploaded object was not found in the bucket. Upload the file first.") from exc


def download_to(key: str, destination: Path) -> None:
    client, bucket = _client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, validate_key(key), str(destination))


def delete_object(key: str) -> None:
    client, bucket = _client()
    client.delete_object(Bucket=bucket, Key=validate_key(key))
