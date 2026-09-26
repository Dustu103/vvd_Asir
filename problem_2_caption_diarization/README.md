# Problem 2: Automated Bengali Subtitle & Closed-Caption Pipeline with Speaker Diarization
Track: Speech & Language / Media Localisation

## Overview
This standalone directory contains the complete end-to-end pipeline for **Problem 2**:
- **Video Ingestion & Shot Cut Detection**: Extracts 16kHz mono audio and detects visual scene cuts.
- **Voice Activity Detection (VAD) & Sound Event Detection (SED)**: Detects dialogue and non-speech events ([MUSIC], [LAUGHTER]).
- **Anti-Hallucination Guard**: Defends against phantom Whisper loops over silence/music (Toughest Test).
- **Persistent Speaker Diarization**: Stable speaker IDs with vocative cast resolution.
- **Broadcast Timed-Text Formatter**: Strict Netflix/BBC standards (CPS <= 17.5, CPL <= 38, max 2 lines).
- **Multilingual Translation**: English (.srt) and Hindi (.srt) subtitle tracks.
- **QC Engine & Ranked Review Queue**: 10-point audit exporting qc_report.json and interactive HTML dashboard.

## Directory Layout
```
problem_2_caption_diarization/
├── src/
│   ├── video_audio.py
│   ├── vad_and_sed.py
│   ├── diarization.py
│   ├── timed_text.py
│   ├── translation.py
│   ├── qc_engine.py
│   └── pipeline.py
├── configs/
│   ├── timed_text.yaml
│   ├── qc_policy.yaml
│   └── cast_registry.yaml
├── notebooks/
│   └── Problem2_Caption_Diarization_Pipeline.ipynb
└── deliverables/
```

## Quick Run
```bash
python -m problem_2_caption_diarization.src.pipeline --video ../data/raw_videos/feluda.mp4
```
