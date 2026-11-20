"""Resolve local model files or download them from private Streamlit secrets URLs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


ASSET_SPECS = {
    "eye_cnn": {
        "filename": "cnn_model.h5",
        "local": Path("models/cnn_model.h5"),
        "secret": "EYE_CNN_URL",
    },
    "mouth_cnn": {
        "filename": "mouth_cnn_model.h5",
        "local": Path("models/mouth_cnn_model.h5"),
        "secret": "MOUTH_CNN_URL",
    },
    "ear_classifier": {
        "filename": "ml_model.pkl",
        "local": Path("models/ml_model.pkl"),
        "secret": "EAR_MODEL_URL",
    },
    "landmark_predictor": {
        "filename": "shape_predictor_68_face_landmarks.dat",
        "local": Path("shape_predictor_68_face_landmarks.dat"),
        "secret": "DLIB_PREDICTOR_URL",
    },
}


def _secret_value(secrets: Any, key: str) -> str | None:
    value = os.environ.get(key)
    if not value and secrets is not None:
        try:
            value = secrets.get(key)
        except Exception:
            value = None
    return str(value).strip() if value else None


def _download_asset(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Asset URLs must be direct HTTPS download URLs.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 300),
            headers={"User-Agent": "driver-drowsiness-streamlit/1.0"},
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        if not temporary.exists() or temporary.stat().st_size == 0:
            raise ValueError("The downloaded asset is empty.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def resolve_model_assets(
    project_root: str | Path, secrets: Any = None
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Return resolved paths, missing asset instructions, and download errors.

    Local files are preferred. When absent, the corresponding *_URL secret is
    downloaded to a URL-specific cache directory that is ignored by Git.
    """
    root = Path(project_root).resolve()
    cache_root = root / ".model_cache"
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    errors: list[str] = []

    for asset_key, spec in ASSET_SPECS.items():
        local_path = root / spec["local"]
        if local_path.is_file() and local_path.stat().st_size > 0:
            resolved[asset_key] = local_path
            continue

        secret_key = spec["secret"]
        url = _secret_value(secrets, secret_key)
        if not url:
            missing.append(f"{spec['filename']} — configure {secret_key}")
            continue

        # A changed URL receives a fresh cache location rather than reusing stale bytes.
        url_fingerprint = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        cached_path = cache_root / url_fingerprint / spec["filename"]
        if not cached_path.is_file() or cached_path.stat().st_size == 0:
            try:
                _download_asset(url, cached_path)
            except Exception as exc:
                # Do not echo the configured URL: it may contain a signed token.
                if isinstance(exc, requests.RequestException):
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    detail = f"HTTP status {status}" if status else type(exc).__name__
                else:
                    detail = type(exc).__name__
                errors.append(
                    f"{spec['filename']}: download failed ({detail}). Check the URL and access permissions."
                )
                continue

        resolved[asset_key] = cached_path

    return resolved, missing, errors
