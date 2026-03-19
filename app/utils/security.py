"""Security utilities: safe file storage and SHA-256 hash verification."""
from __future__ import annotations
import hashlib
import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename


def safe_upload_path(original_filename: str, upload_dir: Path) -> tuple[Path, str]:
    """
    Return a safe absolute path and its UUID-prefixed filename.
    Prevents path traversal and filename collisions.
    """
    ext = Path(secure_filename(original_filename)).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return upload_dir / unique_name, unique_name


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Path, expected_hash: str) -> bool:
    """Return True only if the file matches the expected SHA-256 digest."""
    return sha256_file(path) == expected_hash
