from __future__ import annotations

import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_FILES = 50_000
MAX_UNPACKED_BYTES = 32 * 1024**3


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"Unsafe path in model archive: {name}")
    if ":" in member.parts[0]:
        raise ValueError(f"Unsafe absolute path in model archive: {name}")
    return member


def _find_model_directory(extraction_root: Path) -> Path:
    configs = sorted(extraction_root.rglob("config.json"), key=lambda item: len(item.parts))
    for config in configs:
        parent = config.parent
        if any(parent.glob("*.safetensors")) or any(parent.glob("*.bin")) or any(parent.glob("*.gguf")):
            return parent
    model_files = [
        file for file in extraction_root.rglob("*")
        if file.is_file() and file.suffix.lower() in {".gguf", ".onnx", ".tflite", ".lite"}
    ]
    if model_files:
        shallowest = min(model_files, key=lambda item: len(item.parts))
        return shallowest.parent
    return extraction_root


def unpack_model_archive(archive_path: Path, model_root: Path) -> Path:
    """Safely unpack a local model ZIP inside model_root and return its model directory."""
    if archive_path.suffix.lower() != ".zip" or not archive_path.is_file():
        raise ValueError("Select a local .zip model archive.")
    root = model_root.resolve()
    source = archive_path.resolve()
    if source != root and root not in source.parents:
        raise ValueError("Model archive must be inside the configured local models directory.")

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", archive_path.stem).strip("._") or "model"
    imports_dir = (root / ".imports").resolve()
    if root not in imports_dir.parents:
        raise ValueError("Invalid extraction directory.")
    imports_dir.mkdir(parents=True, exist_ok=True)
    target = imports_dir / safe_stem
    if target.exists():
        return _find_model_directory(target)

    temporary = Path(tempfile.mkdtemp(prefix=f".{safe_stem}-", dir=imports_dir))
    try:
        try:
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                if len(members) > MAX_ARCHIVE_FILES:
                    raise ValueError("Model archive contains too many files.")
                total_bytes = sum(info.file_size for info in members)
                if total_bytes > MAX_UNPACKED_BYTES:
                    raise ValueError("Model archive expands beyond the 32 GiB safety limit.")
                for info in members:
                    relative = _safe_member_path(info.filename)
                    unix_mode = info.external_attr >> 16
                    unix_type = stat.S_IFMT(unix_mode)
                    if unix_type == stat.S_IFLNK or (unix_type and unix_type not in {stat.S_IFREG, stat.S_IFDIR}):
                        raise ValueError(f"Links and special files are not allowed in model archives: {info.filename}")
                    destination = (temporary / Path(*relative.parts)).resolve()
                    if destination != temporary and temporary not in destination.parents:
                        raise ValueError(f"Archive entry escapes extraction directory: {info.filename}")
                    if info.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as input_file, destination.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        except zipfile.BadZipFile as exc:
            raise ValueError("The selected ZIP file is invalid or corrupted.") from exc
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _find_model_directory(target)
