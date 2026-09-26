"""
run_problem2.py
===============
End-to-End Problem 2 Pipeline Runner with Real Neural Models.
Transcribes actual speech with Whisper, segments with VAD,
clusters speakers with Diarization, translates to EN and HI,
and runs a 10-point broadcast QC audit.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem_2_caption_diarization.src.pipeline import BengaliSubtitlePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Problem 2 Pipeline with Real ML Models")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--outdir", default="deliverables", help="Output directory")
    parser.add_argument("--max_dur", type=float, default=60.0, help="Max duration in seconds to process")
    parser.add_argument(
        "--base_model",
        default="SayedShaun/bengali-whisper-medium",
        help="Whisper model ID (default: SayedShaun/bengali-whisper-medium)",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--drive_root", default=str(PROJECT_ROOT), help="Path containing adapters/ directory")
    args = parser.parse_args()

    print("=" * 68)
    print("  Hoichoi Problem 2 — Production Model Pipeline")
    print(f"  Video      : {args.video}")
    print(f"  Max Duration: {args.max_dur}s")
    print(f"  Drive/Root : {args.drive_root}")
    print(f"  ASR Model  : {args.base_model}")
    print(f"  Device     : {args.device}")
    print("=" * 68)

    pipeline = BengaliSubtitlePipeline(
        drive_root=args.drive_root,
        output_dir=args.outdir,
        device=args.device,
        base_model_id=args.base_model,
        use_pyannote=False,
    )

    results = pipeline.run(
        video_path=args.video,
        max_duration_sec=args.max_dur,
    )

    print("\n[RESULT SUMMARY]")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
