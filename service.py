from __future__ import annotations

from dataclasses import dataclass
import re
import time
import uuid

import numpy as np

from .audio import AudioBuffer, StreamingAudioWindow
from .config import SttConfig
from .domain import SttSessionState, TranscriptKind, TranscriptSegment
from .engine import TranscriptAdapter


@dataclass(frozen=True, slots=True)
class STTEmitResult:
    events: list[TranscriptSegment]
    has_more: bool = False


class STTService:
    def __init__(
        self,
        config: SttConfig | None = None,
        adapter: TranscriptAdapter | None = None,
    ) -> None:
        self.config = config or SttConfig.from_env()
        self.adapter = adapter
        self.buffer = AudioBuffer(self.config)
        self.state: SttSessionState | None = None

    def start_session(self, session_id: str | None = None, *, started_at: float | None = None) -> SttSessionState:
        self.buffer.reset()
        self.state = SttSessionState(
            session_id=session_id or str(uuid.uuid4()),
            started_at=started_at if started_at is not None else time.time(),
        )
        return self.state

    def ensure_session(self) -> SttSessionState:
        if self.state is None:
            return self.start_session()
        return self.state

    def ingest_audio(
        self,
        samples: np.ndarray | bytes | bytearray | memoryview,
        *,
        received_at: float | None = None,
    ) -> STTEmitResult:
        state = self.ensure_session()
        now = received_at if received_at is not None else time.time()
        audio = self._coerce_samples(samples)
        if self._is_speech(audio):
            state.last_speech_at = now

        self.buffer.append(audio)

        events: list[TranscriptSegment] = []
        while self.buffer.can_emit_window():
            window = self.buffer.pop_window()
            if window is None:
                break
            partial_text = self._transcribe_window(window)
            partial_event = self._record_partial(state=state, text=partial_text, created_at=now)
            if partial_event is not None:
                events.append(partial_event)

            final_event = self._maybe_commit_final(state=state, created_at=now)
            if final_event is not None:
                events.append(final_event)
                self._reset_utterance_state(state)
                self.buffer.reset()
                return STTEmitResult(events=events, has_more=False)

        final_event = self._maybe_commit_final(state=state, created_at=now)
        if final_event is not None:
            events.append(final_event)
            self._reset_utterance_state(state)
            self.buffer.reset()
            return STTEmitResult(events=events, has_more=False)

        return STTEmitResult(events=events, has_more=self.buffer.can_emit_window())

    def reset_session(self) -> None:
        self.buffer.reset()
        self.state = None

    def _coerce_samples(
        self,
        samples: np.ndarray | bytes | bytearray | memoryview,
    ) -> np.ndarray:
        if isinstance(samples, (bytes, bytearray, memoryview)):
            audio = np.frombuffer(samples, dtype=np.float32)
        else:
            audio = np.asarray(samples, dtype=np.float32)
        return np.ascontiguousarray(audio.reshape(-1))

    def _is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        return rms >= self.config.speech_rms_threshold

    def _transcribe_window(self, window: StreamingAudioWindow) -> str:
        if self.adapter is None:
            return ""
        return self.adapter.transcribe_window(window, self.config).strip()

    def _record_partial(
        self,
        *,
        state: SttSessionState,
        text: str,
        created_at: float,
    ) -> TranscriptSegment | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        partial_key = self._partial_key(cleaned)
        if partial_key == state.last_partial_key:
            state.partial_repeat_count += 1
            return None

        state.last_partial_text = cleaned
        state.last_partial_key = partial_key
        state.partial_repeat_count = 1
        return TranscriptSegment(
            session_id=state.session_id,
            kind=TranscriptKind.PARTIAL,
            text=cleaned,
            created_at=created_at,
            revision=state.next_revision(),
        )

    def _maybe_commit_final(
        self,
        *,
        state: SttSessionState,
        created_at: float,
    ) -> TranscriptSegment | None:
        if not state.last_partial_text:
            return None

        pause_elapsed = state.last_speech_at is not None and (
            created_at - state.last_speech_at >= self.config.pause_timeout_secs
        )
        repeated_partial = state.partial_repeat_count >= self.config.partial_repeat_threshold

        if not pause_elapsed and not repeated_partial:
            return None

        return TranscriptSegment(
            session_id=state.session_id,
            kind=TranscriptKind.FINAL,
            text=state.last_partial_text,
            created_at=created_at,
            revision=state.next_revision(),
        )

    def _reset_utterance_state(self, state: SttSessionState) -> None:
        state.last_speech_at = None
        state.last_partial_text = ""
        state.last_partial_key = ""
        state.partial_repeat_count = 0

    def _partial_key(self, text: str) -> str:
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(normalized.split())
