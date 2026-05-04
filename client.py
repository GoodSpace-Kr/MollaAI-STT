from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time

import sounddevice as sd
from websocket import ABNF, create_connection


def _shorten(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio to the STT websocket.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stt/ws", help="STT websocket URL")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Audio channel count")
    parser.add_argument("--blocksize", type=int, default=1024, help="Frames per audio block")
    parser.add_argument("--encoding", choices=("pcm16", "float32"), default="pcm16")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stop_event = threading.Event()
    audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

    def _request_stop(*_args: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    ws = create_connection(args.url)

    def _recv_loop() -> None:
        while not stop_event.is_set():
            try:
                message = ws.recv()
            except Exception:
                stop_event.set()
                break

            if not message:
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(message)
                continue

            message_type = payload.get("type")
            if message_type in ("partial", "final"):
                text = str(payload.get("text", ""))
                print(f"[{message_type.upper()}] {_shorten(text)}")
            elif message_type == "llm":
                text = str(payload.get("text", ""))
                print(f"[LLM] {_shorten(text)}")
            elif message_type == "tts":
                print(f"[TTS] {payload.get('wav_path', '')}")
            elif message_type == "ready":
                print(payload.get("message", "ready"))
            elif message_type == "started":
                print(f"session started: {payload.get('session_id', '')}")
            elif message_type == "reset":
                print("session reset")
            elif message_type == "error":
                print(f"[ERROR] {payload.get('message', '')}")
            else:
                print(payload)

    recv_thread = threading.Thread(target=_recv_loop, daemon=True)
    recv_thread.start()

    ws.send(
        json.dumps(
            {
                "type": "start",
                "encoding": args.encoding,
                "config": {
                    "sample_rate": args.sample_rate,
                    "channels": args.channels,
                },
            }
        )
    )

    def _audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
        if status:
            print(status)
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            print("audio queue full, dropping block")

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            channels=args.channels,
            dtype="int16" if args.encoding == "pcm16" else "float32",
            blocksize=args.blocksize,
            callback=_audio_callback,
        ):
            print("Speak into the microphone. Ctrl+C to stop.")
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
    finally:
        stop_event.set()
        time.sleep(0.2)
        try:
            ws.close()
        except Exception:
            pass
        recv_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
