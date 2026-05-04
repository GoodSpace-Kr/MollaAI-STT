from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

import numpy as np

from .config import SttConfig
from .domain import AudioChunk


AudioEncoding = Literal["pcm16", "float32"]


def decode_audio_payload(payload: bytes, encoding: AudioEncoding) -> np.ndarray:
    if not payload:
        return np.array([], dtype=np.float32)

    if encoding == "float32":
        return np.frombuffer(payload, dtype=np.float32).astype(np.float32, copy=False)

    if encoding == "pcm16":
        return (np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(f"Unsupported audio encoding: {encoding}")


@dataclass(frozen=True, slots=True)
class StreamingAudioWindow:
    left_context: np.ndarray
    chunk: np.ndarray
    right_context: np.ndarray
    left_padding_samples: int = 0
    right_padding_samples: int = 0

    @property
    def samples(self) -> np.ndarray:
        parts = [self.left_context, self.chunk, self.right_context]
        if not any(part.size for part in parts):
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)


class AudioBuffer:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._frames: Deque[np.ndarray] = deque()
        self._base_sample_index = 0
        self._total_samples = 0
        self._next_emit_sample = 0

    @property
    def chunk_samples(self) -> int:
        return max(1, int(self.config.chunk_secs * self.config.sample_rate))

    @property
    def left_context_samples(self) -> int:
        return max(0, int(self.config.left_context_secs * self.config.sample_rate))

    @property
    def right_context_samples(self) -> int:
        return max(0, int(self.config.right_context_secs * self.config.sample_rate))

    @property
    def available_samples(self) -> int:
        return self._total_samples

    @property
    def available_seconds(self) -> float:
        return self._total_samples / float(self.config.sample_rate)

    def reset(self) -> None:
        self._frames.clear()
        self._base_sample_index = 0
        self._total_samples = 0
        self._next_emit_sample = 0

    def append(self, samples: np.ndarray | bytes | bytearray | memoryview) -> int:
        frame = self._normalize_samples(samples)
        if frame.size == 0:
            return 0

        self._frames.append(frame)
        self._total_samples += int(frame.size)
        return int(frame.size)

    def can_emit_window(self) -> bool:
        buffer_end = self._base_sample_index + self._total_samples
        required = self._next_emit_sample + self.chunk_samples + self.right_context_samples
        return buffer_end >= required

    def pop_window(self) -> StreamingAudioWindow | None:
        if not self.can_emit_window():
            return None

        window = self._build_window(
            chunk_start=self._next_emit_sample,
            chunk_end=self._next_emit_sample + self.chunk_samples,
            pad_right=False,
        )
        self._next_emit_sample += self.chunk_samples
        self._discard_obsolete_prefix()
        return window

    def snapshot_chunk(self, start_index: int, end_index: int) -> AudioChunk:
        audio = self._slice_absolute(start_index, end_index)
        return AudioChunk(
            session_id="",
            chunk_index=0,
            samples=audio.tobytes(),
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            created_at=0.0,
        )

    def _normalize_samples(self, samples: np.ndarray | bytes | bytearray | memoryview) -> np.ndarray:
        if isinstance(samples, (bytes, bytearray, memoryview)):
            frame = np.frombuffer(samples, dtype=np.float32)
        else:
            frame = np.asarray(samples, dtype=np.float32)
        return np.ascontiguousarray(frame.reshape(-1))

    def _build_window(
        self,
        *,
        chunk_start: int,
        chunk_end: int,
        pad_right: bool,
    ) -> StreamingAudioWindow:
        left_start = max(0, chunk_start - self.left_context_samples)
        left_context = self._slice_absolute(left_start, chunk_start)
        chunk = self._slice_absolute(chunk_start, chunk_end)

        if pad_right:
            right_context = np.array([], dtype=np.float32)
            right_padding_samples = max(0, self.right_context_samples - max(0, self._total_samples - chunk_end))
            if right_padding_samples:
                right_context = np.zeros(right_padding_samples, dtype=np.float32)
        else:
            right_end = chunk_end + self.right_context_samples
            right_context = self._slice_absolute(chunk_end, right_end)
            right_padding_samples = max(0, self.right_context_samples - int(right_context.size))
            if right_padding_samples:
                right_context = np.concatenate(
                    [right_context, np.zeros(right_padding_samples, dtype=np.float32)]
                )

        left_padding_samples = max(0, self.left_context_samples - int(left_context.size))
        if left_padding_samples:
            left_context = np.concatenate([np.zeros(left_padding_samples, dtype=np.float32), left_context])

        return StreamingAudioWindow(
            left_context=left_context,
            chunk=chunk,
            right_context=right_context,
            left_padding_samples=left_padding_samples,
            right_padding_samples=right_padding_samples,
        )

    def _slice_absolute(self, start: int, end: int) -> np.ndarray:
        if end <= start:
            return np.array([], dtype=np.float32)

        relative_start = max(0, start - self._base_sample_index)
        relative_end = max(0, end - self._base_sample_index)
        if relative_start >= self._total_samples:
            return np.array([], dtype=np.float32)

        relative_end = min(relative_end, self._total_samples)
        if relative_end <= relative_start:
            return np.array([], dtype=np.float32)

        parts: list[np.ndarray] = []
        current_index = 0
        wanted_start = relative_start
        wanted_end = relative_end

        for frame in self._frames:
            frame_end = current_index + int(frame.size)
            if frame_end <= wanted_start:
                current_index = frame_end
                continue
            if current_index >= wanted_end:
                break

            local_start = max(0, wanted_start - current_index)
            local_end = min(int(frame.size), wanted_end - current_index)
            if local_end > local_start:
                parts.append(frame[local_start:local_end])

            current_index = frame_end

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32, copy=False)

    def _discard_obsolete_prefix(self) -> None:
        keep_from = max(0, self._next_emit_sample - self.left_context_samples)
        if keep_from <= self._base_sample_index:
            return

        drop_count = keep_from - self._base_sample_index
        while self._frames and drop_count > 0:
            frame = self._frames[0]
            if drop_count >= int(frame.size):
                drop_count -= int(frame.size)
                self._base_sample_index += int(frame.size)
                self._total_samples -= int(frame.size)
                self._frames.popleft()
                continue

            self._frames[0] = frame[drop_count:]
            self._base_sample_index += drop_count
            self._total_samples -= drop_count
            drop_count = 0
