#!/usr/bin/env python
"""Receive ESP32 JPEG captures and append their classifications to a CSV log."""

from __future__ import annotations

import argparse
import csv
import hmac
import json
import math
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
DEFAULT_IMAGE_DIR = ROOT / "capImages"
DEFAULT_LOG_PATH = ROOT / "detections.csv"
DEFAULT_MOTION_LOG_PATH = ROOT / "motion_events.csv"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_MOTION_EVENT_BYTES = 1024
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CSV_FIELDS = [
    "event_id",
    "captured_at",
    "image_path",
    "species",
    "confidence",
    "model_version",
    "is_alert",
    "is_target_species",
    "is_human",
    "trigger_source",
    "camera_sensor",
    "pir_gpio",
    "inference_ms",
    "uptime_ms",
    "frame_width",
    "frame_height",
    "image_bytes",
    "scores",
]
MOTION_CSV_FIELDS = ["event_id", "captured_at", "uptime_ms", "pir_gpio", "trigger_source"]
MEDIUM_THREAT_CLASSES = {"wild_boar", "nilgai"}
HIGH_THREAT_CLASSES = {"leopard", "tiger", "hyena"}
LOG_LOCK = Lock()


def get_threat_level(metadata: dict[str, Any]) -> str:
    species = metadata["species"]
    if species == "human" or not metadata["is_alert"]:
        return "low"
    if species in HIGH_THREAT_CLASSES:
        return "high"
    if species in MEDIUM_THREAT_CLASSES:
        return "medium"
    return "low"


def validate_metadata(raw: str) -> dict[str, Any]:
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    required = {
        "event_id",
        "captured_at",
        "species",
        "confidence",
        "model_version",
        "is_alert",
        "is_target_species",
        "is_human",
        "trigger_source",
        "camera_sensor",
        "pir_gpio",
        "inference_ms",
        "uptime_ms",
        "frame_width",
        "frame_height",
        "scores",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"missing metadata fields: {', '.join(sorted(missing))}")
    if not isinstance(metadata["event_id"], str) or not EVENT_ID_PATTERN.fullmatch(
        metadata["event_id"]
    ):
        raise ValueError("event_id contains unsupported characters")
    if not isinstance(metadata["species"], str) or len(metadata["species"]) > 64:
        raise ValueError("species must be a string of at most 64 characters")
    if not isinstance(metadata["model_version"], str) or len(metadata["model_version"]) > 32:
        raise ValueError("model_version must be a string of at most 32 characters")

    confidence = metadata["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
        raise ValueError("confidence must be numeric")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    for key in ("is_alert", "is_target_species", "is_human"):
        if not isinstance(metadata[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("pir_gpio", "inference_ms", "uptime_ms", "frame_width", "frame_height"):
        if isinstance(metadata[key], bool) or not isinstance(metadata[key], int) or metadata[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not isinstance(metadata["scores"], dict):
        raise ValueError("scores must be a JSON object")
    for name, score in metadata["scores"].items():
        if (
            not isinstance(name, str)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("scores must map class names to probabilities from 0 to 1")
    for key in ("captured_at", "trigger_source", "camera_sensor"):
        if not isinstance(metadata[key], str) or len(metadata[key]) > 64:
            raise ValueError(f"{key} must be a string of at most 64 characters")
    return metadata


def validate_capture_metadata(raw: str) -> dict[str, Any]:
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    required = {
        "event_id",
        "captured_at",
        "uptime_ms",
        "frame_width",
        "frame_height",
        "pir_gpio",
        "trigger_source",
        "camera_sensor",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"missing metadata fields: {', '.join(sorted(missing))}")
    if not isinstance(metadata["event_id"], str) or not EVENT_ID_PATTERN.fullmatch(
        metadata["event_id"]
    ):
        raise ValueError("event_id contains unsupported characters")
    for key in ("uptime_ms", "frame_width", "frame_height", "pir_gpio"):
        if (
            isinstance(metadata[key], bool)
            or not isinstance(metadata[key], int)
            or metadata[key] < 0
        ):
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("captured_at", "trigger_source", "camera_sensor"):
        if not isinstance(metadata[key], str) or len(metadata[key]) > 64:
            raise ValueError(f"{key} must be a string of at most 64 characters")
    if metadata["frame_width"] == 0 or metadata["frame_height"] == 0:
        raise ValueError("frame dimensions must be positive")
    return metadata


def validate_motion_metadata(raw: str) -> dict[str, Any]:
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    required = {"event_id", "captured_at", "uptime_ms", "pir_gpio", "trigger_source"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"missing metadata fields: {', '.join(sorted(missing))}")
    if not isinstance(metadata["event_id"], str) or not EVENT_ID_PATTERN.fullmatch(
        metadata["event_id"]
    ):
        raise ValueError("event_id contains unsupported characters")
    for key in ("uptime_ms", "pir_gpio"):
        if (
            isinstance(metadata[key], bool)
            or not isinstance(metadata[key], int)
            or metadata[key] < 0
        ):
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("captured_at", "trigger_source"):
        if not isinstance(metadata[key], str) or len(metadata[key]) > 64:
            raise ValueError(f"{key} must be a string of at most 64 characters")
    return metadata


class CaptureRequestHandler(BaseHTTPRequestHandler):
    server_version = "SDG15Capture/1.0"

    def __init__(
        self,
        *args: Any,
        image_dir: Path,
        log_path: Path,
        motion_log_path: Path,
        classifier: Any | None,
        token: str,
        **kwargs: Any,
    ) -> None:
        self.image_dir = image_dir
        self.log_path = log_path
        self.motion_log_path = motion_log_path
        self.classifier = classifier
        self.token = token
        super().__init__(*args, **kwargs)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format_string % args}")

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/dashboard"):
            try:
                dashboard = (ROOT / "dashboard.html").read_bytes()
            except OSError as exc:
                self._send_json(500, {"error": f"could not read dashboard: {exc}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(dashboard)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(dashboard)
            return
        if path == "/api/latest":
            try:
                latest_capture = self._latest_capture()
            except (OSError, ValueError, csv.Error) as exc:
                self._send_json(500, {"error": f"could not read latest capture: {exc}"})
                return
            self._send_json(200, {"capture": latest_capture})
            return
        if path.startswith("/images/"):
            event_id = path.removeprefix("/images/").removesuffix(".jpg")
            if not EVENT_ID_PATTERN.fullmatch(event_id) or path != f"/images/{event_id}.jpg":
                self._send_json(404, {"error": "not found"})
                return
            image_path = self.image_dir / f"{event_id}.jpg"
            try:
                image = image_path.read_bytes()
            except FileNotFoundError:
                self._send_json(404, {"error": "image not found"})
                return
            except OSError as exc:
                self._send_json(500, {"error": f"could not read image: {exc}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(image)
            return
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def _latest_capture(self) -> dict[str, Any] | None:
        if not self.log_path.is_file():
            return None

        with self.log_path.open(newline="", encoding="utf-8") as log_file:
            last_row = None
            for last_row in csv.DictReader(log_file):
                pass
        if last_row is None:
            return None

        event_id = last_row["event_id"]
        if not EVENT_ID_PATTERN.fullmatch(event_id):
            raise ValueError("latest log row contains an invalid event_id")
        scores_field = next(
            (
                field
                for field in last_row
                if isinstance(field, str)
                and (field == "scores" or field.startswith("scores"))
            ),
            None,
        )
        return {
            "event_id": event_id,
            "captured_at": last_row["captured_at"],
            "image_url": f"/images/{event_id}.jpg",
            "species": last_row["species"],
            "confidence": float(last_row["confidence"]),
            "model_version": last_row["model_version"],
            "is_alert": last_row["is_alert"].lower() == "true",
            "is_target_species": last_row["is_target_species"].lower() == "true",
            "is_human": last_row["is_human"].lower() == "true",
            "inference_ms": int(last_row["inference_ms"]),
            "frame_width": int(last_row["frame_width"]),
            "frame_height": int(last_row["frame_height"]),
            "scores": json.loads(last_row.get(scores_field) or "{}"),
        }

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/motion":
            self._handle_motion_alert()
            return
        if path != "/capture":
            self._send_json(404, {"error": "not found"})
            return
        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(authorization, expected):
            self._send_json(401, {"error": "invalid upload token"})
            return
        if self.headers.get_content_type() != "image/jpeg":
            self._send_json(415, {"error": "expected image/jpeg"})
            return

        length_header = self.headers.get("Content-Length")
        try:
            content_length = int(length_header) if length_header is not None else 0
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "empty image body"})
            return
        if content_length > MAX_IMAGE_BYTES:
            self._send_json(413, {"error": "image exceeds 5 MiB limit"})
            return

        image = self.rfile.read(content_length)
        if len(image) != content_length:
            self._send_json(400, {"error": "incomplete image body"})
            return

        raw_capture = self.headers.get("X-Capture")
        if raw_capture is not None:
            if self.classifier is None:
                self._send_json(503, {"error": "host classifier is not configured"})
                return
            try:
                capture_metadata = validate_capture_metadata(raw_capture)
                prediction = self.classifier.classify(image)
                metadata = validate_metadata(json.dumps({**capture_metadata, **prediction}))
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except (OSError, RuntimeError) as exc:
                self._send_json(500, {"error": f"host classification failed: {exc}"})
                return
        else:
            raw_metadata = self.headers.get("X-Detection")
            if raw_metadata is None:
                self._send_json(400, {"error": "missing X-Capture metadata"})
                return
            try:
                metadata = validate_metadata(raw_metadata)
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
                return

        image_name = f"{metadata['event_id']}.jpg"
        image_path = self.image_dir / image_name
        with LOG_LOCK:
            if image_path.exists():
                self._send_json(409, {"error": "event_id already exists"})
                return

            try:
                self.image_dir.mkdir(parents=True, exist_ok=True)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = image_path.with_suffix(".jpg.tmp")
                temporary_path.write_bytes(image)
                os.replace(temporary_path, image_path)
                row = {
                    "event_id": metadata["event_id"],
                    "captured_at": metadata["captured_at"],
                    "image_path": f"capImages/{image_name}",
                    "species": metadata["species"],
                    "confidence": f"{metadata['confidence']:.6f}",
                    "model_version": metadata["model_version"],
                    "is_alert": metadata["is_alert"],
                    "is_target_species": metadata["is_target_species"],
                    "is_human": metadata["is_human"],
                    "trigger_source": metadata["trigger_source"],
                    "camera_sensor": metadata["camera_sensor"],
                    "pir_gpio": metadata["pir_gpio"],
                    "inference_ms": metadata["inference_ms"],
                    "uptime_ms": metadata["uptime_ms"],
                    "frame_width": metadata["frame_width"],
                    "frame_height": metadata["frame_height"],
                    "image_bytes": len(image),
                    "scores": json.dumps(metadata["scores"], separators=(",", ":")),
                }
                write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
                with self.log_path.open("a", newline="", encoding="utf-8") as log_file:
                    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except OSError as exc:
                self._send_json(500, {"error": f"could not persist capture: {exc}"})
                return

        self._send_json(
            201,
            {
                "event_id": metadata["event_id"],
                "image": f"capImages/{image_name}",
                "threat_level": get_threat_level(metadata),
            },
        )

    def _handle_motion_alert(self) -> None:
        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(authorization, expected):
            self._send_json(401, {"error": "invalid upload token"})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "expected application/json"})
            return

        length_header = self.headers.get("Content-Length")
        try:
            content_length = int(length_header) if length_header is not None else 0
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "empty motion alert"})
            return
        if content_length > MAX_MOTION_EVENT_BYTES:
            self._send_json(413, {"error": "motion alert metadata exceeds size limit"})
            return

        raw_metadata = self.rfile.read(content_length)
        if len(raw_metadata) != content_length:
            self._send_json(400, {"error": "incomplete motion alert"})
            return
        try:
            metadata = validate_motion_metadata(raw_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        with LOG_LOCK:
            try:
                if self.motion_log_path.exists():
                    with self.motion_log_path.open(newline="", encoding="utf-8") as log_file:
                        if any(
                            row["event_id"] == metadata["event_id"]
                            for row in csv.DictReader(log_file)
                        ):
                            self._send_json(409, {"error": "event_id already exists"})
                            return

                self.motion_log_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = (
                    not self.motion_log_path.exists()
                    or self.motion_log_path.stat().st_size == 0
                )
                with self.motion_log_path.open("a", newline="", encoding="utf-8") as log_file:
                    writer = csv.DictWriter(log_file, fieldnames=MOTION_CSV_FIELDS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metadata)
            except OSError as exc:
                self._send_json(500, {"error": f"could not persist motion alert: {exc}"})
                return

        self._send_json(201, {"event_id": metadata["event_id"]})


def make_server(
    host: str,
    port: int,
    token: str,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
    motion_log_path: Path | None = None,
    classifier: Any | None = None,
) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("an upload token is required")

    def handler(*args: Any, **kwargs: Any) -> CaptureRequestHandler:
        return CaptureRequestHandler(
            *args,
            image_dir=image_dir,
            log_path=log_path,
            motion_log_path=motion_log_path or log_path.with_name("motion_events.csv"),
            classifier=classifier,
            token=token,
            **kwargs,
        )

    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token", default=os.environ.get("SDG15_UPLOAD_TOKEN", ""))
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "ml" / "models" / "best_model.pt",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        from hardware.host_inference import WildlifeClassifier

        classifier = WildlifeClassifier(args.checkpoint, args.device)
        server = make_server(
            args.host,
            args.port,
            args.token,
            args.image_dir,
            args.log_path,
            classifier=classifier,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Capture receiver listening on {args.host}:{args.port}")
    print(f"JPEG directory: {args.image_dir.resolve()}")
    print(f"Detection log: {args.log_path.resolve()}")
    print(f"Classifier checkpoint: {args.checkpoint.resolve()} ({classifier.device})")
    print(f"Classification latency on host: model-dependent")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping capture receiver")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
