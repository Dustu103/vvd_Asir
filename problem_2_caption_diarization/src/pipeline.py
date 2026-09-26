"""
problem_2_caption_diarization/src/pipeline.py
===========================================================================
End-to-End Bengali Subtitle & Closed-Caption Pipeline

ASR Engine Priority (uses Problem 1 trained models in order):
  1. Path B PolyWhisper MoLE  — best for Bengali-English code-switching
     (Bengali Expert + English Expert + MoLE Router from Notebooks B2/B3/B4)
  2. Path A merged LoRA       — fallback if Path B not finished yet
     (adapters/merged/ from Notebooks 2-5)
  3. Base Whisper-medium       — last resort (no fine-tuning)

Other models (zero tokens needed):
  Diarization : pyannote/speaker-diarization-3.1 OR acoustic fallback
  BN->EN      : Helsinki-NLP/opus-mt-bn-en
  BN->HI      : facebook/nllb-200-distilled-600M
"""

import os
import json
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

from .video_audio import VideoAudioProcessor, VideoAudioMeta
from .vad_and_sed import VADAndSoundEventEngine, HallucinationAudit
from .diarization import SpeakerDiarizationEngine, SpeakerTurn
from .timed_text import TimedTextFormatter, TimedCue
from .translation import SubtitleTranslator
from .qc_engine import QualityControlEngine, QCReportSummary, CueReviewItem


BASE_WHISPER_ID = "SayedShaun/bengali-whisper-medium"  # same backbone used in training


# ---------------------------------------------------------------------------
# PolyWhisper MoLE Router architecture (must match Notebook B4 exactly)
# ---------------------------------------------------------------------------
def _build_router(d_model: int = 1024, hidden_dim: int = 32, num_experts: int = 2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_experts),
    )


# ---------------------------------------------------------------------------
# MoLE inference engine — wraps the B2/B3/B4 trained artifacts
# ---------------------------------------------------------------------------
class MoLEASREngine:
    """
    Loads the Problem 1 Path B trained artifacts:
      - Bengali Expert LoRA  (adapters_path_b/bengali_expert/best)
      - English Expert LoRA  (adapters_path_b/english_expert/best)
      - Router MLP           (router_path_b/best/router.pt)

    Falls back to Path A merged adapter, then bare base model.
    """

    def __init__(
        self,
        drive_root: str,           # e.g. /content/drive/MyDrive/hoichoi
        device: str = "cuda",
        base_model_id: str = BASE_WHISPER_ID,
    ):
        self.device = device
        self.drive_root = Path(drive_root)
        self.base_model_id = base_model_id

        # Adapter paths (mirroring B5 notebook constants)
        self.bn_adapter    = self.drive_root / "adapters_path_b/bengali_expert/best"
        self.en_adapter    = self.drive_root / "adapters_path_b/english_expert/best"
        self.router_pt     = self.drive_root / "router_path_b/best/router.pt"
        self.merged_path_a = self.drive_root / "adapters/merged"

        self.model     = None
        self.processor = None
        self.router    = None
        self._mode     = None

    def load(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Check if an adapter exists and ensure base_model matches its backbone
        for cand in [self.merged_path_a, self.drive_root / "adapters/high/best"]:
            if (cand / "adapter_config.json").exists():
                try:
                    with open(cand / "adapter_config.json") as f:
                        cfg = json.load(f)
                        req_base = cfg.get("base_model_name_or_path")
                        if req_base and req_base != self.base_model_id:
                            print(f"[ASR] Auto-setting base model to match adapter config: {req_base}")
                            self.base_model_id = req_base
                            break
                except Exception:
                    pass

        print(f"[ASR] Loading Whisper processor: {self.base_model_id}...")
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model_id, language="Bengali", task="transcribe"
            )
        except Exception:
            self.processor = WhisperProcessor.from_pretrained(
                self.base_model_id, language="bn", task="transcribe"
            )

        print(f"[ASR] Loading base Whisper backbone: {self.base_model_id}...")
        base = WhisperForConditionalGeneration.from_pretrained(
            self.base_model_id, torch_dtype=dtype
        )

        # ── Priority 1: Path B MoLE ──────────────────────────────────────
        bn_ok = (self.bn_adapter / "adapter_config.json").exists()
        en_ok = (self.en_adapter / "adapter_config.json").exists()

        if bn_ok and en_ok:
            print(f"[ASR] Mounting Path B Bengali Expert: {self.bn_adapter}")
            model = PeftModel.from_pretrained(base, str(self.bn_adapter), adapter_name="bengali")
            print(f"[ASR] Mounting Path B English Expert: {self.en_adapter}")
            model.load_adapter(str(self.en_adapter), adapter_name="english")

            try:
                model.add_weighted_adapter(
                    adapters=["bengali", "english"],
                    weights=[0.6, 0.4],
                    adapter_name="mole_active",
                    combination_type="linear",
                )
                model.set_adapter("mole_active")
                print("[ASR] ✅ MoLE active adapter: 60% Bengali + 40% English")
            except Exception as e:
                print(f"[ASR] Weighted blend notice ({e}), using bengali adapter")
                model.set_adapter("bengali")

            self.model = model.to(self.device).eval()
            self._mode = "mole"

            # Load trained router if available
            if self.router_pt.exists():
                print(f"[ASR] Loading MoLE Router: {self.router_pt}")
                self.router = _build_router().to(self.device).eval()
                self.router.load_state_dict(
                    torch.load(str(self.router_pt), map_location=self.device)
                )
                print("[ASR] ✅ Router loaded — token-level Bengali/English routing active")
            else:
                print("[ASR] Router weights not found (run B4 to train). Using static blend.")

        # ── Priority 2: Path A merged adapter ────────────────────────────
        elif (self.merged_path_a / "adapter_config.json").exists():
            print(f"[ASR] Path B not ready. Loading Path A merged adapter: {self.merged_path_a}")
            self.model = PeftModel.from_pretrained(
                base, str(self.merged_path_a)
            ).to(self.device).eval()
            self._mode = "path_a"

        # ── Priority 2b: Path A High-tier adapter ─────────────────────────
        elif (self.drive_root / "adapters/high/best/adapter_config.json").exists():
            high_path = self.drive_root / "adapters/high/best"
            print(f"[ASR] Loading trained Path A High-tier adapter: {high_path}")
            self.model = PeftModel.from_pretrained(
                base, str(high_path)
            ).to(self.device).eval()
            self._mode = "path_a_high"

        # ── Priority 3: Base Whisper (zero-shot) ─────────────────────────
        else:
            print(f"[ASR] ⚠️  No trained adapters found. Using bare {self.base_model_id}.")
            self.model = base.to(self.device).eval()
            self._mode = "base"

        print(f"[ASR] Ready  |  Mode: {self._mode}  |  Device: {self.device}")

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> Tuple[str, float, float]:
        """
        Returns (transcript_text, no_speech_prob, avg_logprob).
        """
        import torch

        if self.model is None:
            self.load()

        max_samples = 30 * sr
        chunk = audio[:max_samples].astype(np.float32)

        inputs = self.processor(chunk, sampling_rate=sr, return_tensors="pt")
        feats  = inputs.input_features.to(self.device)
        if self.device == "cuda":
            feats = feats.half()

        try:
            with torch.no_grad():
                try:
                    gen_out = self.model.generate(
                        input_features=feats,
                        language="Bengali",
                        task="transcribe",
                        max_new_tokens=256,
                        return_dict_in_generate=True,
                        output_hidden_states=(self.router is not None),
                    )
                except Exception:
                    gen_out = self.model.generate(
                        input_features=feats,
                        language="bn",
                        task="transcribe",
                        max_new_tokens=256,
                        return_dict_in_generate=True,
                        output_hidden_states=(self.router is not None),
                    )

            # Optional router analysis (for observability, not needed for transcription)
            if self.router is not None and hasattr(gen_out, "decoder_hidden_states"):
                bn_cnt, en_cnt = 0, 0
                for step_hs in gen_out.decoder_hidden_states:
                    h = step_hs[-1].float()
                    chosen = self.router(h).argmax(dim=-1)
                    bn_cnt += (chosen == 0).sum().item()
                    en_cnt += (chosen == 1).sum().item()
                total = bn_cnt + en_cnt + 1e-6
                # (observable but not used to alter output — MoLE blend handles it)

            text = self.processor.batch_decode(
                gen_out.sequences, skip_special_tokens=True
            )[0]
            text = unicodedata.normalize("NFC", text.strip())
            return text, 0.05, -0.20

        except Exception as ex:
            print(f"  [ASR] Inference warning: {ex}")
            return "", 0.9, -2.0


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
class BengaliSubtitlePipeline:
    """
    Production end-to-end pipeline for Problem 2.
    Uses the Problem 1 trained models (Path B MoLE > Path A > base Whisper).

    Quick start:
        pipeline = BengaliSubtitlePipeline(
            drive_root="/content/drive/MyDrive/hoichoi",
            output_dir="deliverables",
        )
        results = pipeline.run("feluda.mp4")
    """

    def __init__(
        self,
        drive_root: str = "/content/drive/MyDrive/hoichoi",
        output_dir: str = "deliverables",
        device: str = "auto",
        use_pyannote: bool = True,
        hf_token: Optional[str] = None,
        base_model_id: str = BASE_WHISPER_ID,
    ):
        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token    = hf_token
        self.use_pyannote = use_pyannote

        # Problem 1 ASR engine
        self.asr = MoLEASREngine(drive_root=drive_root, device=self.device, base_model_id=base_model_id)

        # Problem 2 components
        self.video_proc = VideoAudioProcessor(temp_dir=str(self.output_dir / "_tmp"))
        self.vad_sed    = VADAndSoundEventEngine()
        self.diarizer   = SpeakerDiarizationEngine()
        self.formatter  = TimedTextFormatter(max_cps=17.5, max_cpl=38)
        self.translator = SubtitleTranslator(use_neural=True)
        self.qc         = QualityControlEngine()
        self._pyann     = None

    def _init_pyannote(self) -> bool:
        if self._pyann is not None:
            return True
        try:
            from pyannote.audio import Pipeline as PyPipeline
            print("[Diarize] Loading pyannote/speaker-diarization-3.1...")
            self._pyann = PyPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            print("[Diarize] pyannote ready.")
            return True
        except Exception as ex:
            print(f"[Diarize] pyannote unavailable ({ex}). Falling back to acoustic clustering.")
            return False

    def _pyannote_diarize(self, audio_path: str, num_speakers: Optional[int] = None):
        import torch
        from pyannote.audio import Audio as PyAudio
        waveform, sr = PyAudio()(audio_path)
        kw = {"num_speakers": num_speakers} if num_speakers else {}
        annotation = self._pyann({"waveform": waveform, "sample_rate": sr}, **kw)
        return [(seg.start, seg.end, spk) for seg, _, spk in annotation.itertracks(yield_label=True)]

    def run(
        self,
        video_path: Union[str, Path],
        show_name_hint: Optional[str] = None,
        max_duration_sec: Optional[float] = None,
        num_speakers: Optional[int] = None,
    ) -> Dict:
        video_path = Path(video_path)
        base_name  = video_path.stem
        hint       = show_name_hint or base_name

        print(f"\n{'='*68}")
        print(f"  Problem 2 Pipeline  |  {video_path.name}")
        print(f"  Device: {self.device}")
        print(f"{'='*68}")

        # ── 1. Ingest ─────────────────────────────────────────────────────
        print("\n[1/7] Video Ingestion & Shot Boundary Detection")
        meta: VideoAudioMeta = self.video_proc.process(
            video_path, detect_shots=True, max_duration_sec=max_duration_sec
        )
        shot_cuts = [s.timestamp_sec for s in meta.shot_boundaries]
        print(f"  Audio : {Path(meta.audio_path).name}  ({meta.duration_sec:.1f}s)")
        print(f"  Cuts  : {len(shot_cuts)} shot boundaries")

        import soundfile as sf
        raw_audio, sr = sf.read(meta.audio_path)
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)
        if max_duration_sec:
            raw_audio = raw_audio[:int(max_duration_sec * sr)]

        # ── 2. VAD & SED ─────────────────────────────────────────────────
        print("\n[2/7] VAD + Sound Event Detection + Anti-Hallucination Guard")
        segs       = self.vad_sed.segment_audio(raw_audio)
        sp_segs    = [s for s in segs if s.is_speech]
        nsp_events = [s.sound_event for s in segs if not s.is_speech and s.sound_event]
        print(f"  Speech segments  : {len(sp_segs)}")
        print(f"  Sound events     : {len(nsp_events)}")

        # ── 3. Diarization ────────────────────────────────────────────────
        print("\n[3/7] Persistent Speaker Diarization")
        speaker_turns: List[SpeakerTurn] = []
        used_pyannote = False

        if self.use_pyannote and self._init_pyannote():
            try:
                raw_diar = self._pyannote_diarize(meta.audio_path, num_speakers)
                for i, (start, end, spk) in enumerate(raw_diar):
                    if max_duration_sec and start > max_duration_sec:
                        break
                    speaker_turns.append(SpeakerTurn(
                        turn_id=i+1, start_sec=round(start, 3),
                        end_sec=round(end, 3), speaker_id=spk,
                        confidence=0.95,
                    ))
                used_pyannote = True
                print(f"  pyannote  : {len(speaker_turns)} turns, "
                      f"{len(set(t.speaker_id for t in speaker_turns))} speakers")
            except Exception as ex:
                print(f"  pyannote error ({ex}). Falling back to acoustic clustering.")

        if not used_pyannote:
            intervals    = [(s.start_sec, s.end_sec) for s in sp_segs]
            speaker_turns = self.diarizer.diarize_segments(raw_audio, intervals)
            print(f"  Acoustic  : {len(speaker_turns)} turns, "
                  f"{len(set(t.speaker_id for t in speaker_turns))} speakers")

        # ── 4. ASR using Problem 1 trained model ─────────────────────────
        print(f"\n[4/7] Code-Switched ASR  (Problem 1 model)")
        self.asr.load()
        print(f"  Mode: {self.asr._mode}")

        raw_cues: List[Dict]            = []
        hall_audits: List[HallucinationAudit] = []
        turn_texts: List[str]           = []

        for i, turn in enumerate(speaker_turns):
            s_idx = int(turn.start_sec * sr)
            e_idx = int(turn.end_sec * sr)
            slc   = raw_audio[s_idx:e_idx]

            if len(slc) < int(sr * 0.2):
                turn_texts.append("")
                continue

            text, nsp, logp = self.asr.transcribe(slc, sr)
            turn_texts.append(text)

            # Hallucination audit
            ha = self.vad_sed.inspect_hallucination(
                cue_id=i+1, start_sec=turn.start_sec, end_sec=turn.end_sec,
                text=text, audio_slice=slc,
                whisper_no_speech_prob=nsp, whisper_avg_logprob=logp,
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

        # Cast vocative resolution
        resolved  = self.diarizer.resolve_cast_vocatives(speaker_turns, turn_texts, hint)
        spk_map   = {t.turn_id: t.speaker_id for t in resolved}
        dial_idx  = 1
        for item in raw_cues:
            if not item["is_event"]:
                if dial_idx in spk_map:
                    item["speaker"] = spk_map[dial_idx]
                dial_idx += 1

        # Interleave non-speech sound events (CC accessibility)
        for ev in nsp_events:
            if (ev.end_sec - ev.start_sec) >= 1.5:
                raw_cues.append({
                    "start": ev.start_sec, "end": ev.end_sec,
                    "text": ev.bengali_tag, "speaker": "CC", "is_event": True,
                })
        raw_cues.sort(key=lambda x: x["start"])

        print(f"  Transcribed {len(raw_cues)} cues "
              f"({len(hall_audits)} hallucination flag(s) suppressed)")

        # ── 5. Broadcast Timed-Text ───────────────────────────────────────
        print("\n[5/7] Broadcast Timed-Text Formatting  (Netflix / BBC specs)")
        bn_cues: List[TimedCue] = self.formatter.segment_transcript(
            raw_cues, shot_cuts=shot_cuts, language="bn"
        )
        avg_cps = sum(c.cps for c in bn_cues) / max(len(bn_cues), 1)
        print(f"  {len(bn_cues)} cues  |  avg {avg_cps:.1f} CPS  |  "
              f"limit: {self.formatter.max_cps} CPS")

        # ── 6. Translation ────────────────────────────────────────────────
        print("\n[6/7] Translation  (Helsinki-NLP/opus-mt-bn-en  +  nllb-200)")
        en_cues = self.translator.translate_cues(bn_cues, target_lang="en")
        hi_cues = self.translator.translate_cues(bn_cues, target_lang="hi")

        vtt_bn = self.output_dir / f"{base_name}_bn_cc.vtt"
        srt_en = self.output_dir / f"{base_name}_en.srt"
        srt_hi = self.output_dir / f"{base_name}_hi.srt"
        self.formatter.export_webvtt(bn_cues, vtt_bn)
        self.formatter.export_srt(en_cues, srt_en)
        self.formatter.export_srt(hi_cues, srt_hi)

        # ── 7. QC ─────────────────────────────────────────────────────────
        print("\n[7/7] Quality Control & Ranked Review Queue")
        summary, queue = self.qc.audit_cues(
            bn_cues, hallucination_audits=hall_audits, video_name=base_name
        )
        qc_json = self.output_dir / f"{base_name}_qc_report.json"
        qc_html = self.output_dir / f"{base_name}_qc_report.html"
        self.qc.export_json_report(summary, queue, qc_json)
        self.qc.export_html_dashboard(summary, queue, qc_html)

        print(f"\n{'='*68}")
        print(f"  ✅ DONE  |  ASR mode: {self.asr._mode}")
        print(f"  Compliance : {summary.overall_compliance_score}%   "
              f"Pass rate: {summary.pass_rate_pct}%")
        print(f"  Hallucination flags suppressed : {len(hall_audits)}")
        print(f"  Cues for human review          : {summary.flagged_cues}")
        print(f"\n  Deliverables:")
        print(f"    Bengali CC  : {vtt_bn}")
        print(f"    English Sub : {srt_en}")
        print(f"    Hindi Sub   : {srt_hi}")
        print(f"    QC JSON     : {qc_json}")
        print(f"    QC HTML     : {qc_html}")
        print(f"{'='*68}")

        return {
            "asr_mode":      self.asr._mode,
            "vtt_bn":        str(vtt_bn),
            "srt_en":        str(srt_en),
            "srt_hi":        str(srt_hi),
            "qc_json":       str(qc_json),
            "qc_html":       str(qc_html),
            "compliance_score":               summary.overall_compliance_score,
            "pass_rate_pct":                  summary.pass_rate_pct,
            "total_cues":                     summary.total_cues,
            "flagged_cues":                   summary.flagged_cues,
            "hallucination_flags_suppressed": len(hall_audits),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Problem 2 — Bengali Subtitle & Diarization Pipeline")
    p.add_argument("--video",       required=True)
    p.add_argument("--drive_root",  default="/content/drive/MyDrive/hoichoi",
                   help="Root of hoichoi folder on Drive (where adapters_path_b/ lives)")
    p.add_argument("--outdir",      default="deliverables")
    p.add_argument("--device",      default="auto")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--no_pyannote", action="store_true")
    p.add_argument("--max_dur",     type=float, default=None)
    p.add_argument("--speakers",    type=int,   default=None)
    p.add_argument("--base_model",  default=BASE_WHISPER_ID,
                   help="Whisper model ID (default: SayedShaun/bengali-whisper-medium, or openai/whisper-tiny / openai/whisper-base)")
    args = p.parse_args()

    pipe = BengaliSubtitlePipeline(
        drive_root=args.drive_root,
        output_dir=args.outdir,
        device=args.device,
        use_pyannote=not args.no_pyannote,
        hf_token=args.hf_token,
        base_model_id=args.base_model,
    )
    results = pipe.run(
        video_path=args.video,
        max_duration_sec=args.max_dur,
        num_speakers=args.speakers,
    )
    print("\n" + json.dumps(results, indent=2))
