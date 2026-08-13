from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from mobocapture.io import sha256_file


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
OMDET_TURBO_MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"
OMDET_TURBO_REVISION = "b73ddc1becb89eadf37f26deb30c5efe703d6680"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
YUNET_FACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FACE_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def huggingface_cache_root() -> Path:
    """Shared cache used by the pinned open-vocabulary vision models."""

    return model_cache_root() / "huggingface"


def model_cache_root() -> Path:
    configured = os.environ.get("MOBOCAPTURE_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "MoboCapture" / "models"
    return Path.home() / ".cache" / "mobocapture" / "models"


def ensure_hand_landmarker_model() -> tuple[Path, str, str]:
    """Return a verified local model, downloading the pinned official asset once."""

    explicit = os.environ.get("MOBOCAPTURE_HAND_MODEL")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MOBOCAPTURE_HAND_MODEL does not exist: {path}")
        return path, sha256_file(path), "environment_override"

    destination = model_cache_root() / "mediapipe" / "hand_landmarker-float16-v1.task"
    if destination.is_file():
        actual_hash = sha256_file(destination)
        if actual_hash == HAND_LANDMARKER_SHA256:
            return destination, actual_hash, HAND_LANDMARKER_URL
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        HAND_LANDMARKER_URL,
        headers={"User-Agent": "MoboCapture/0.1 model-fetcher"},
    )
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != HAND_LANDMARKER_SHA256:
            raise RuntimeError(
                "Downloaded hand model failed SHA-256 verification: "
                f"expected {HAND_LANDMARKER_SHA256}, got {actual_hash}"
            )
        shutil.move(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, HAND_LANDMARKER_SHA256, HAND_LANDMARKER_URL


def ensure_yunet_face_model() -> tuple[Path, str, str]:
    destination = model_cache_root() / "opencv" / "face_detection_yunet_2023mar.onnx"
    if destination.is_file() and sha256_file(destination) == YUNET_FACE_SHA256:
        return destination, YUNET_FACE_SHA256, YUNET_FACE_URL
    destination.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.download")
    request = urllib.request.Request(YUNET_FACE_URL, headers={"User-Agent": "MoboCapture/0.1"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual = digest.hexdigest()
        if actual != YUNET_FACE_SHA256:
            raise RuntimeError(
                f"Downloaded YuNet model failed SHA-256 verification: {actual}"
            )
        shutil.move(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, YUNET_FACE_SHA256, YUNET_FACE_URL
