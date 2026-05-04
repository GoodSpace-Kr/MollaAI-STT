from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if value.startswith(("'", '"')):
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            return key, value[1:]
        return key, value[1:end]

    if "#" in value:
        value = value.split("#", 1)[0].rstrip()

    return key, value


def _load_dotenv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [Path.cwd() / ".env", project_root / ".env"]

    seen: set[Path] = set()
    for env_path in candidate_paths:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue

            key, value = parsed
            os.environ.setdefault(key, value)


_load_dotenv()


@dataclass(frozen=True, slots=True)
class SttConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_secs: float = 2.0
    left_context_secs: float = 10.0
    right_context_secs: float = 2.0
    batch_size: int = 1
    model_name: str = ""
    model_path: str = ""
    language: str = "en"
    use_timestamps: bool = False
    speech_rms_threshold: float = 0.01
    pause_timeout_secs: float = 2.0
    partial_repeat_threshold: int = 2

    @classmethod
    def from_env(cls) -> SttConfig:
        return cls(
            sample_rate=int(os.getenv("STT_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("STT_CHANNELS", "1")),
            chunk_secs=float(os.getenv("STT_CHUNK_SECS", "2.0")),
            left_context_secs=float(os.getenv("STT_LEFT_CONTEXT_SECS", "10.0")),
            right_context_secs=float(os.getenv("STT_RIGHT_CONTEXT_SECS", "2.0")),
            batch_size=int(os.getenv("STT_BATCH_SIZE", "1")),
            model_name=os.getenv("STT_MODEL_NAME", ""),
            model_path=os.getenv("STT_MODEL_PATH", ""),
            language=os.getenv("STT_LANGUAGE", "en"),
            use_timestamps=os.getenv("STT_USE_TIMESTAMPS", "0") == "1",
            speech_rms_threshold=float(os.getenv("STT_SPEECH_RMS_THRESHOLD", "0.01")),
            pause_timeout_secs=float(os.getenv("STT_PAUSE_TIMEOUT_SECS", "2.0")),
            partial_repeat_threshold=int(os.getenv("STT_PARTIAL_REPEAT_THRESHOLD", "2")),
        )
