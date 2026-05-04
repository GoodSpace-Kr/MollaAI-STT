from __future__ import annotations

from dataclasses import asdict, replace
import importlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from threading import Lock
from typing import Any, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def _load_local_package() -> str:
    package_name = "_molla_stt_runtime"
    if package_name in sys.modules:
        return package_name

    package_root = Path(__file__).resolve().parent
    spec = spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load STT package from local workspace.")

    module = module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return package_name


_PKG_NAME = _load_local_package()
_PKG = sys.modules[_PKG_NAME]
_AUDIO_MODULE = importlib.import_module(f"{_PKG_NAME}.audio")

NemoAsrAdapter = getattr(_PKG, "NemoAsrAdapter")
STTService = getattr(_PKG, "STTService")
SttConfig = getattr(_PKG, "SttConfig")
decode_audio_payload = getattr(_AUDIO_MODULE, "decode_audio_payload")


class AdapterRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._adapters: dict[tuple[str, str], Any] = {}

    def get(self, config: Any) -> Any:
        key = (str(config.model_name), str(config.model_path))
        with self._lock:
            adapter = self._adapters.get(key)
            if adapter is None:
                adapter = NemoAsrAdapter(config)
                self._adapters[key] = adapter
            return adapter


adapter_registry = AdapterRegistry()
app = FastAPI(title="Molla STT")


def _merge_config(payload: dict[str, Any]) -> Any:
    base = SttConfig.from_env()
    allowed_keys = set(asdict(base).keys())
    overrides = {key: value for key, value in payload.items() if key in allowed_keys}

    if "use_timestamps" in overrides:
        overrides["use_timestamps"] = _coerce_bool(overrides["use_timestamps"])

    return replace(base, **overrides)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "type": event.kind.value,
        "session_id": event.session_id,
        "text": event.text,
        "revision": event.revision,
        "created_at": event.created_at,
        "confidence": event.confidence,
    }


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/stt/ws")
async def stt_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "ready", "message": "stt websocket ready"})

    service: Any | None = None
    encoding = "pcm16"

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                break

            text_data = cast(str | None, message.get("text"))
            bytes_data = cast(bytes | None, message.get("bytes"))

            if text_data is not None:
                service, encoding = await _handle_text_message(
                    websocket=websocket,
                    service=service,
                    encoding=encoding,
                    raw_message=text_data,
                )
                continue

            if bytes_data is not None:
                if service is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Send a start message before audio bytes."}
                    )
                    continue

                samples = decode_audio_payload(bytes_data, encoding)
                result = service.ingest_audio(samples)
                for event in result.events:
                    await websocket.send_json(_event_payload(event))
                continue

    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)


async def _handle_text_message(
    *,
    websocket: WebSocket,
    service: Any | None,
    encoding: str,
    raw_message: str,
) -> tuple[Any | None, str]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "Invalid JSON message."})
        return service, encoding

    command = str(payload.get("type", "")).strip().lower()

    if command == "start":
        config = _merge_config(payload.get("config", {}))
        next_encoding = str(payload.get("encoding", "pcm16")).strip().lower() or "pcm16"
        if next_encoding not in {"pcm16", "float32"}:
            await websocket.send_json({"type": "error", "message": f"Unsupported encoding: {next_encoding}"})
            return service, encoding

        adapter = adapter_registry.get(config)
        next_service = STTService(config=config, adapter=adapter)
        state = next_service.start_session(session_id=payload.get("session_id"))

        await websocket.send_json(
            {
                "type": "started",
                "session_id": state.session_id,
                "config": asdict(config),
            }
        )
        return next_service, next_encoding

    if command == "reset":
        if service is not None:
            service.reset_session()
        await websocket.send_json({"type": "reset"})
        return None, encoding

    if command == "ping":
        await websocket.send_json({"type": "pong"})
        return service, encoding

    await websocket.send_json({"type": "error", "message": f"Unsupported message type: {command or 'unknown'}"})
    return service, encoding
