"""Transcribe videos locally with faster-whisper into Scribe-compatible JSON.

The output shape is intentionally compatible with pack_transcripts.py,
timeline_view.py, and render.py:

    <edit_dir>/transcripts/<video_stem>.json

Usage:
    python helpers/transcribe_local.py <video_path> --language fr
    python helpers/transcribe_local.py <videos_dir> --language fr --model small
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_videos(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix not in VIDEO_EXTS:
            sys.exit(f"not a supported video file: {path}")
        return [path]
    if path.is_dir():
        videos = sorted(child for child in path.iterdir() if child.is_file() and child.suffix in VIDEO_EXTS)
        if not videos:
            sys.exit(f"no videos found in {path}")
        return videos
    sys.exit(f"path not found: {path}")


def load_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed. Run: uv add faster-whisper")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_one(
    *,
    model: Any,
    video: Path,
    edit_dir: Path,
    model_name: str,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    force: bool,
) -> Path:
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists() and not force:
        print(f"cached: {out_path.name}")
        return out_path

    print(f"local transcribe: {video.name}", flush=True)
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio_path)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
        )
        segments = list(segments_iter)

    payload = build_scribe_compatible_payload(
        segments=segments,
        info=info,
        model_name=model_name,
        language=language,
        elapsed_seconds=time.time() - started,
    )
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved: {out_path} ({len(payload['words'])} word/spacing entries)")
    return out_path


def build_scribe_compatible_payload(
    *,
    segments: list[Any],
    info: Any,
    model_name: str,
    language: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    plain_text_parts: list[str] = []
    previous_end: float | None = None

    for segment in segments:
        segment_text = str(getattr(segment, "text", "") or "").strip()
        if segment_text:
            plain_text_parts.append(segment_text)
        for word in getattr(segment, "words", None) or []:
            raw_text = str(getattr(word, "word", "") or "").strip()
            start = _float_or_none(getattr(word, "start", None))
            end = _float_or_none(getattr(word, "end", None))
            if not raw_text or start is None:
                continue
            if end is None or end < start:
                end = start

            if previous_end is not None and start > previous_end:
                words.append(
                    {
                        "type": "spacing",
                        "text": " ",
                        "start": round(previous_end, 3),
                        "end": round(start, 3),
                        "speaker_id": "speaker_0",
                    }
                )
            words.append(
                {
                    "type": "word",
                    "text": raw_text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker_id": "speaker_0",
                }
            )
            previous_end = end

    detected_language = getattr(info, "language", None) or language
    language_probability = _float_or_none(getattr(info, "language_probability", None))
    duration = _float_or_none(getattr(info, "duration", None))
    return {
        "text": " ".join(plain_text_parts).strip(),
        "language_code": detected_language,
        "language_probability": language_probability,
        "duration": duration,
        "words": words,
        "metadata": {
            "provider": "faster-whisper-local",
            "model": model_name,
            "word_timestamps": True,
            "diarize": False,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_edit_dir(source: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    if source.is_dir():
        return (source / "edit").resolve()
    return (source.parent / "edit").resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local faster-whisper transcription for video-use")
    ap.add_argument("source", type=Path, help="Video file or directory containing videos")
    ap.add_argument("--edit-dir", type=Path, default=None, help="Default: <source_dir>/edit")
    ap.add_argument("--model", default="small", help="faster-whisper model name. Default: small")
    ap.add_argument("--language", default=None, help="Optional language code, e.g. fr or en")
    ap.add_argument("--device", default="auto", help="faster-whisper device. Default: auto")
    ap.add_argument("--compute-type", default="int8", help="Default: int8")
    ap.add_argument("--beam-size", type=int, default=5, help="Default: 5")
    ap.add_argument("--vad-filter", action="store_true", help="Enable faster-whisper VAD filter")
    ap.add_argument("--force", action="store_true", help="Overwrite existing cached transcript JSON")
    args = ap.parse_args()

    source = args.source.resolve()
    videos = find_videos(source)
    edit_dir = default_edit_dir(source, args.edit_dir)
    edit_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading faster-whisper model: {args.model} ({args.device}, {args.compute_type})", flush=True)
    model = load_model(args.model, args.device, args.compute_type)

    outputs = [
        transcribe_one(
            model=model,
            video=video,
            edit_dir=edit_dir,
            model_name=args.model,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            force=args.force,
        )
        for video in videos
    ]
    print(f"done: {len(outputs)} transcript(s) in {edit_dir / 'transcripts'}")


if __name__ == "__main__":
    main()
