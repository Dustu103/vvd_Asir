"""
test_pipeline_docker.py
=======================
End-to-end Problem 2 pipeline test — NO model downloads.

All neural model loaders are stubbed so every module falls through
to its spectral / lexical fallback path.

Usage inside container:
    python test_pipeline_docker.py --video /video/test.mp4 --outdir /out [--max_dur 60]
"""

import sys
import os
import types
import argparse
import time

# ──────────────────────────────────────────────────────────────
# 1. Stub all heavy ML packages BEFORE any project imports
#    Use proper stub modules (not None) to avoid Python's import guard
# ──────────────────────────────────────────────────────────────

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__    = None
    m.__loader__  = None
    m.__package__ = name
    m.__path__    = []            # marks it as a package
    return m

# torch — provide just enough surface so `import torch` works
# and `torch.cuda.is_available()` returns False
_torch = _make_stub("torch")
_torch_cuda = _make_stub("torch.cuda")
_torch_cuda.is_available = lambda: False
_torch.cuda = _torch_cuda
_torch.hub  = _make_stub("torch.hub")
_torch.hub.load = staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("stub")))
_torch.from_numpy  = None
_torch.no_grad     = staticmethod(lambda: __import__("contextlib").nullcontext())
_torch.nn          = _make_stub("torch.nn")
_torch.nn.functional = _make_stub("torch.nn.functional")

for _sub in [
    "torch.cuda", "torch.hub", "torch.nn",
    "torch.nn.functional", "torch.utils",
]:
    sys.modules.setdefault(_sub, _make_stub(_sub))
sys.modules["torch"] = _torch

# Everything else → plain empty stub
_STUBS = [
    "torchaudio",
    "transformers",
    "peft",
    "peft.import_utils",
    "speechbrain",
    "speechbrain.pretrained",
    "panns_inference",
    "pyannote",
    "pyannote.audio",
    "librosa",           # use soundfile directly for audio read
]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_stub(_pkg)

# librosa.resample stub (used by PANNsSED — which will fail and fall back)
_librosa_stub = sys.modules["librosa"]
_librosa_stub.resample = staticmethod(lambda y, orig_sr, target_sr: y)
_librosa_stub.load = staticmethod(lambda path, sr=None, mono=True, **kw: (__import__("soundfile").read(path, always_2d=False)[0], sr or 16000))

# ──────────────────────────────────────────────────────────────
# 2. Project root on path
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────
# 3. Import pipeline modules (each falls to its fallback quietly)
# ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  Hoichoi Problem 2 — Docker Fallback Test")
print("  (spectral VAD + spectral SED + spectral diarization)")
print("=" * 65)

from problem_2_caption_diarization.src.video_audio import VideoAudioProcessor
from problem_2_caption_diarization.src.vad_and_sed  import VADAndSoundEventEngine
from problem_2_caption_diarization.src.diarization  import SpeakerDiarizationEngine
from problem_2_caption_diarization.src.timed_text   import TimedTextFormatter
from problem_2_caption_diarization.src.translation  import SubtitleTranslator
from problem_2_caption_diarization.src.qc_engine    import QualityControlEngine

import numpy as np
import soundfile as sf

# ──────────────────────────────────────────────────────────────
# 4. Args
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video",   required=True)
parser.add_argument("--outdir",  default="/out")
parser.add_argument("--max_dur", type=float, default=60.0)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
base_name = os.path.splitext(os.path.basename(args.video))[0]
t0 = time.time()

# ──────────────────────────────────────────────────────────────
# STAGE 1 — Video ingestion & shot boundary detection
# ──────────────────────────────────────────────────────────────
print(f"\n[1/7] Ingesting: {args.video}")
proc = VideoAudioProcessor(temp_dir=os.path.join(args.outdir, "_tmp"))
meta = proc.process(args.video, detect_shots=True, max_duration_sec=args.max_dur)
print(f"      Audio   : {meta.audio_path}  ({meta.duration_sec:.1f}s)")
print(f"      Cuts    : {len(meta.shot_boundaries)} shot boundaries detected")

audio, sr = sf.read(meta.audio_path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio[:int(args.max_dur * sr)]

# ──────────────────────────────────────────────────────────────
# STAGE 2 — VAD + SED  (spectral fallback — no Silero / PANNs)
# ──────────────────────────────────────────────────────────────
print("\n[2/7] VAD + SED  (spectral fallback)")
vad    = VADAndSoundEventEngine(sample_rate=sr)
segs   = vad.segment_audio(audio)
sp_segs   = [s for s in segs if s.is_speech]
nsp_events = [s.sound_event for s in segs if not s.is_speech and s.sound_event]
print(f"      Speech segments : {len(sp_segs)}")
print(f"      Sound events    : {len(nsp_events)}")
for ev in nsp_events[:4]:
    print(f"        [{ev.start_sec:.1f}→{ev.end_sec:.1f}s] {ev.english_tag}  /  {ev.bengali_tag}")

# ──────────────────────────────────────────────────────────────
# STAGE 3 — Diarization  (spectral fingerprint — no ECAPA-TDNN)
# ──────────────────────────────────────────────────────────────
print("\n[3/7] Diarization  (spectral fingerprint fallback)")
diarizer  = SpeakerDiarizationEngine(sample_rate=sr)
intervals = [(s.start_sec, s.end_sec) for s in sp_segs]
turns     = diarizer.diarize_segments(audio, intervals)
spk_ids   = sorted(set(t.speaker_id for t in turns))
print(f"      Turns    : {len(turns)}")
print(f"      Speakers : {spk_ids}")
for t in turns[:4]:
    print(f"        {t.speaker_id}  [{t.start_sec:.1f}→{t.end_sec:.1f}s]")

# ──────────────────────────────────────────────────────────────
# STAGE 4 — Synthetic Bengali dialogue (no Whisper in docker)
# ──────────────────────────────────────────────────────────────
print("\n[4/7] ASR  (Bengali dialogue stubs — no model in docker test)")
DEMO = [
    "ওর office-এ একটা urgent meeting আছে, জলদি পৌঁছাতে হবে।",
    "তোপসে, ক্যামেরাটা ঠিক করো — মগনলাল শেঠ সাবধানী লোক।",
    "ওসি হারুন সাহেব থানায় অপেক্ষা করছেন, ফাইলটা নিয়ে চলো।",
    "মান্দারের তীরে আজ বড় কোনো ঘটনা ঘটতে চলেছে।",
    "দাদা, এই খাবারের স্বাদ কিন্তু অসাধারণ হয়েছে!",
    "আমাদের কাছে আর বেশি সময় নেই, সিদ্ধান্ত এখনই নিতে হবে।",
    "সে কি সত্যিই জানে না, নাকি আমাদের বোকা বানাচ্ছে?",
    "এই case-এর behind একটা বড় conspiracy আছে।",
]

hall_audits, raw_cues = [], []
for i, turn in enumerate(turns):
    s_idx = int(turn.start_sec * sr)
    e_idx = int(turn.end_sec   * sr)
    slc   = audio[s_idx:e_idx]
    text  = DEMO[i % len(DEMO)]

    ha = vad.inspect_hallucination(
        cue_id=i + 1, start_sec=turn.start_sec, end_sec=turn.end_sec,
        text=text, audio_slice=slc,
        whisper_no_speech_prob=0.05, whisper_avg_logprob=-0.20,
    )
    if ha:
        hall_audits.append(ha)
        if ha.recommended_action == "SUPPRESS":
            text = ha.replacement_event or "[SILENCE]"

    if text:
        raw_cues.append({
            "start": turn.start_sec, "end": turn.end_sec,
            "text": text, "speaker": turn.speaker_id, "is_event": False,
        })

for ev in nsp_events:
    if (ev.end_sec - ev.start_sec) >= 1.5:
        raw_cues.append({
            "start": ev.start_sec, "end": ev.end_sec,
            "text": ev.bengali_tag, "speaker": "CC", "is_event": True,
        })

raw_cues.sort(key=lambda x: x["start"])
print(f"      {len(raw_cues)} cues  |  {len(hall_audits)} hallucination flag(s)")

# ──────────────────────────────────────────────────────────────
# STAGE 5 — Broadcast timed-text formatting
# ──────────────────────────────────────────────────────────────
print("\n[5/7] Broadcast Timed-Text Formatting  (Netflix/BBC specs)")
shot_cuts = [s.timestamp_sec for s in meta.shot_boundaries]
formatter = TimedTextFormatter(max_cps=17.5, max_cpl=38)
bn_cues   = formatter.segment_transcript(raw_cues, shot_cuts=shot_cuts, language="bn")
avg_cps   = sum(c.cps for c in bn_cues) / max(len(bn_cues), 1)
print(f"      {len(bn_cues)} broadcast cues  |  avg {avg_cps:.1f} CPS  |  limit 17.5 CPS")

# ──────────────────────────────────────────────────────────────
# STAGE 6 — Translation  (lexical fallback — no neural models)
# ──────────────────────────────────────────────────────────────
print("\n[6/7] Translation  (lexical fallback — BN→EN + BN→HI)")
translator = SubtitleTranslator(use_neural=False)
en_cues    = translator.translate_cues(bn_cues, target_lang="en")
hi_cues    = translator.translate_cues(bn_cues, target_lang="hi")

vtt_bn = os.path.join(args.outdir, f"{base_name}_bn_cc.vtt")
srt_en = os.path.join(args.outdir, f"{base_name}_en.srt")
srt_hi = os.path.join(args.outdir, f"{base_name}_hi.srt")
formatter.export_webvtt(bn_cues, vtt_bn)
formatter.export_srt(en_cues,   srt_en)
formatter.export_srt(hi_cues,   srt_hi)

# ──────────────────────────────────────────────────────────────
# STAGE 7 — QC audit + ranked review queue
# ──────────────────────────────────────────────────────────────
print("\n[7/7] Quality Control Audit & Ranked Review Queue")
qc = QualityControlEngine()
summary, queue = qc.audit_cues(bn_cues, hallucination_audits=hall_audits, video_name=base_name)

qc_json = os.path.join(args.outdir, f"{base_name}_qc_report.json")
qc_html = os.path.join(args.outdir, f"{base_name}_qc_report.html")
qc.export_json_report(summary, queue, qc_json)
qc.export_html_dashboard(summary, queue, qc_html)

elapsed = round(time.time() - t0, 1)

# ──────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ✅  DOCKER TEST COMPLETE  ({elapsed}s)")
print(f"{'='*65}")
print(f"  Video processed   : {args.video}  (first {args.max_dur:.0f}s)")
print(f"  Shot boundaries   : {len(meta.shot_boundaries)}")
print(f"  Speakers detected : {len(set(t.speaker_id for t in turns))}")
print(f"  Total cues        : {summary.total_cues}")
print(f"  Compliance score  : {summary.overall_compliance_score}%")
print(f"  Pass rate         : {summary.pass_rate_pct}%")
print(f"  Hallucination flags suppressed : {len(hall_audits)}")
print(f"\n  Deliverables  →  {args.outdir}/")
for f in [vtt_bn, srt_en, srt_hi, qc_json, qc_html]:
    size = os.path.getsize(f)
    print(f"    {os.path.basename(f):40s}  {size:>6} bytes")

print(f"\n{'─'*65}")
print("  Bengali CC — first 6 cues:")
for cue in bn_cues[:6]:
    txt = cue.text.replace("\n", " | ")
    print(f"    {cue.start_sec:6.2f}→{cue.end_sec:.2f}s  [{cue.speaker_id:<12}]  {txt}")

print(f"\n  English subtitles — first 6 cues:")
for cue in en_cues[:6]:
    txt = cue.text.replace("\n", " | ")
    print(f"    {cue.start_sec:6.2f}→{cue.end_sec:.2f}s  {txt}")

print(f"\n  Hindi subtitles — first 6 cues:")
for cue in hi_cues[:6]:
    txt = cue.text.replace("\n", " | ")
    print(f"    {cue.start_sec:6.2f}→{cue.end_sec:.2f}s  {txt}")
print(f"{'='*65}")
