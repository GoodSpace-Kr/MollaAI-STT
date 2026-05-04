from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .audio import StreamingAudioWindow
from .config import SttConfig


class TranscriptAdapter(Protocol):
    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str: ...


class NemoAsrAdapter:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._model = self._load_model()

    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str:
        audio = np.asarray(window.samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""

        with tempfile.TemporaryDirectory(prefix="nemo_stt_") as tmpdir:
            wav_path = Path(tmpdir) / "window.wav"
            self._write_wav(wav_path, audio, config.sample_rate, config.channels)
            result = self._model.transcribe(
                [str(wav_path)],
                batch_size=max(1, config.batch_size),
            )

        return self._normalize_transcript(result)

    def _load_model(self) -> Any:
        import nemo.collections.asr as nemo_asr

        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"STT_MODEL_PATH not found: {path}")

            model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path))
        elif self.config.model_name:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model_name)
        else:
            raise ValueError("STT_MODEL_PATH 또는 STT_MODEL_NAME 중 하나는 필요합니다.")

        if hasattr(model, "eval"):
            model.eval()

        return model

    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(max(1, channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

    def _normalize_transcript(self, result: Any) -> str:
        text = self._extract_text(result)
        return text.strip()

    def _extract_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            for key in ("text", "transcript", "prediction"):
                nested = self._extract_text(value.get(key))
                if nested.strip():
                    return nested
            return ""

        if isinstance(value, (list, tuple)):
            for item in value:
                nested = self._extract_text(item)
                if nested.strip():
                    return nested
            return ""

        if hasattr(value, "text"):
            return self._extract_text(getattr(value, "text"))

        return ""
