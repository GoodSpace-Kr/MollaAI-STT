from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TranscriptKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    session_id: str
    chunk_index: int
    samples: bytes
    sample_rate: int
    channels: int
    created_at: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    session_id: str
    kind: TranscriptKind
    text: str
    created_at: float
    revision: int = 0
    confidence: float | None = None


@dataclass(slots=True)
class SttSessionState:
    session_id: str
    started_at: float
    last_speech_at: float | None = None
    last_partial_text: str = ""
    last_partial_key: str = ""
    partial_repeat_count: int = 0
    chunk_index: int = 0
    revision: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def next_chunk_index(self) -> int:
        self.chunk_index += 1
        return self.chunk_index

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision
