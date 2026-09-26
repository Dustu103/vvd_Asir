"""
problem_2_caption_diarization/src/vad_and_sed.py
=================================================
Voice Activity Detection, Sound Event Detection & Anti-Hallucination Guard

Real models used (all open-source, no tokens):
  VAD : Silero-VAD   — torch.hub, ONNX, 30ms frame, CPU-friendly
  SED : PANNs CNN14  — panns-inference, 527-class AudioSet tagger
  Fallback: pure spectral heuristics if models unavailable at runtime
"""

import math
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SoundEvent:
    event_type: str           # "MUSIC", "LAUGHTER", "APPLAUSE", "GUNSHOT", etc.
    start_sec: float
    end_sec: float
    confidence: float
    bengali_tag: str
    english_tag: str
    hindi_tag: str


@dataclass
class SpeechSegment:
    segment_id: int
    start_sec: float
    end_sec: float
    speech_prob: float
    energy_dbfs: float
    is_speech: bool
    sound_event: Optional[SoundEvent] = None


@dataclass
class HallucinationAudit:
    cue_id: int
    start_sec: float
    end_sec: float
    hallucinated_text: str
    risk_score: float
    risk_reasons: List[str]
    recommended_action: str           # "SUPPRESS" | "FLAG_FOR_HUMAN_REVIEW"
    replacement_event: Optional[str] = None


# ---------------------------------------------------------------------------
# Bilingual event tag lexicon
# ---------------------------------------------------------------------------
EVENT_LEXICON = {
    "MUSIC":      {"bn": "[মিউজিক]",         "en": "[MUSIC PLAYING]",  "hi": "[संगीत]"},
    "LAUGHTER":   {"bn": "[হাসি]",            "en": "[LAUGHTER]",       "hi": "[हंसी]"},
    "APPLAUSE":   {"bn": "[হাততালি]",         "en": "[APPLAUSE]",       "hi": "[तालियां]"},
    "GUNSHOT":    {"bn": "[গুলির শব্দ]",      "en": "[GUNSHOT]",        "hi": "[गोली की आवाज]"},
    "PHONE_RING": {"bn": "[ফোনের রিং]",       "en": "[PHONE RINGING]",  "hi": "[फोन की घंटी]"},
    "FOOTSTEPS":  {"bn": "[পায়ের আওয়াজ]",   "en": "[FOOTSTEPS]",      "hi": "[कदमों की आहट]"},
    "SILENCE":    {"bn": "[নীরবতা]",          "en": "[SILENCE]",        "hi": "[सन्नाटा]"},
    "SPEECH":     {"bn": "[কথা বলছে]",        "en": "[SPEECH]",         "hi": "[भाषण]"},
    "CROWD":      {"bn": "[ভিড়ের শব্দ]",     "en": "[CROWD NOISE]",    "hi": "[भीड़ शोर]"},
    "VEHICLE":    {"bn": "[গাড়ির শব্দ]",     "en": "[VEHICLE SOUND]",  "hi": "[वाहन की आवाज]"},
    "DOOR":       {"bn": "[দরজার শব্দ]",      "en": "[DOOR SOUND]",     "hi": "[दरवाजे की आवाज]"},
    "RAIN":       {"bn": "[বৃষ্টির শব্দ]",    "en": "[RAIN]",           "hi": "[बारिश]"},
}

# PANNs AudioSet class index → our event type (top 527 AudioSet classes)
PANNS_CLASS_MAP: Dict[int, str] = {
    # Music
    137: "MUSIC", 138: "MUSIC", 140: "MUSIC", 141: "MUSIC",
    # Laughter
    16:  "LAUGHTER", 17: "LAUGHTER",
    # Applause
    388: "APPLAUSE",
    # Gunshot / explosion
    427: "GUNSHOT", 426: "GUNSHOT",
    # Phone
    402: "PHONE_RING", 403: "PHONE_RING",
    # Footsteps
    376: "FOOTSTEPS",
    # Crowd / cheer
    73: "CROWD", 74: "CROWD",
    # Vehicle
    300: "VEHICLE", 301: "VEHICLE",
    # Rain
    497: "RAIN",
    # Silence  (no class — detected by energy threshold)
}


# ---------------------------------------------------------------------------
# VAD — Silero-VAD
# ---------------------------------------------------------------------------
class SileroVAD:
    """
    Wraps the Silero-VAD model from torch.hub.
    Repository: snakers4/silero-vad (no token needed)
    Yields per-30ms frame speech probability.
    """
    SAMPLE_RATE = 16000
    FRAME_MS    = 32         # Silero works at 32ms

    def __init__(self):
        self._model   = None
        self._utils   = None
        self._loaded  = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch
            print("[VAD] Loading Silero-VAD from torch.hub...")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            print("[VAD] Silero-VAD ready.")
        except Exception as ex:
            print(f"[VAD] Silero-VAD unavailable ({ex}). Using spectral fallback.")

    def get_speech_probs(self, audio: np.ndarray, sr: int = 16000) -> List[float]:
        """Returns per-chunk speech probability list (one value per 32ms frame)."""
        self._load()
        if self._model is None:
            return self._spectral_fallback(audio, sr)

        import torch
        try:
            get_speech_timestamps, _, _, _, collect_chunks = self._utils
            t = torch.from_numpy(audio.astype(np.float32))

            # get_speech_timestamps returns segments, not frame-level probs.
            # We use the model directly for frame-level probabilities.
            window = int(sr * self.FRAME_MS / 1000)
            probs = []
            self._model.reset_states()

            for i in range(0, len(t), window):
                chunk = t[i: i + window]
                if len(chunk) < window:
                    chunk = torch.nn.functional.pad(chunk, (0, window - len(chunk)))
                with torch.no_grad():
                    p = self._model(chunk, sr).item()
                probs.append(float(p))

            return probs
        except Exception as ex:
            print(f"[VAD] Silero inference error ({ex}). Using spectral fallback.")
            return self._spectral_fallback(audio, sr)

    @staticmethod
    def _spectral_fallback(audio: np.ndarray, sr: int, frame_ms: int = 32) -> List[float]:
        """Pure numpy fallback when Silero unavailable."""
        frame_n = int(sr * frame_ms / 1000)
        probs = []
        for i in range(0, len(audio), frame_n):
            chunk = audio[i: i + frame_n]
            if len(chunk) == 0:
                continue
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
            rms_db = 20 * np.log10(rms + 1e-12)
            fft = np.abs(np.fft.rfft(chunk)) + 1e-12
            flatness = float(np.exp(np.mean(np.log(fft))) / (np.mean(fft) + 1e-12))
            zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
            score = 0.0
            if rms_db > -42.0:  score += 0.5
            if flatness < 0.40: score += 0.3
            if 0.02 < zcr < 0.35: score += 0.2
            probs.append(score)
        return probs


# ---------------------------------------------------------------------------
# SED — PANNs CNN14
# ---------------------------------------------------------------------------
class PANNsSED:
    """
    Wraps PANNs CNN14 for AudioSet 527-class sound event detection.
    Install: pip install panns-inference
    No token needed. Weights auto-downloaded from Zenodo.
    """

    def __init__(self, device: str = "cpu"):
        self._at   = None
        self._dev  = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from panns_inference import AudioTagging
            print("[SED] Loading PANNs CNN14 (AudioSet 527-class)...")
            self._at = AudioTagging(checkpoint_path=None, device=self._dev)
            print("[SED] PANNs CNN14 ready.")
        except Exception as ex:
            print(f"[SED] PANNs unavailable ({ex}). Using spectral SED fallback.")

    def classify(self, audio: np.ndarray, sr: int = 16000) -> Optional[str]:
        """
        Returns the best-matching EVENT_LEXICON event type for the audio clip,
        or None if confidence < 0.15.
        """
        self._load()

        if self._at is not None:
            try:
                import torch
                # PANNs expects (batch, samples) @ 32kHz mono
                import librosa
                clip = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=32000)
                clip_t = torch.from_numpy(clip[np.newaxis, :])

                with torch.no_grad():
                    _, clipwise_output = self._at.inference(clip_t)
                    probs = clipwise_output[0].cpu().numpy()

                # Walk the PANNS_CLASS_MAP to find the highest matching class
                best_evt, best_prob = None, 0.15  # confidence threshold
                for cls_idx, evt_type in PANNS_CLASS_MAP.items():
                    if cls_idx < len(probs) and probs[cls_idx] > best_prob:
                        best_prob = probs[cls_idx]
                        best_evt = evt_type
                return best_evt

            except Exception as ex:
                print(f"[SED] PANNs inference error ({ex}). Using spectral fallback.")

        return self._spectral_fallback(audio)

    @staticmethod
    def _spectral_fallback(audio: np.ndarray) -> str:
        """Spectral heuristic fallback when PANNs is unavailable."""
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        rms_db = 20 * np.log10(rms + 1e-12)
        if rms_db < -48.0:
            return "SILENCE"
        fft = np.abs(np.fft.rfft(audio)) + 1e-12
        flatness = float(np.exp(np.mean(np.log(fft))) / (np.mean(fft) + 1e-12))
        n = len(fft)
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        if flatness < 0.25:
            return "MUSIC"
        if rms_db > -25.0 and zcr < 0.08 and n > 0:
            return "GUNSHOT"
        return "MUSIC"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class VADAndSoundEventEngine:
    """
    Combines Silero-VAD + PANNs CNN14 + Anti-Hallucination Guard.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.sample_rate       = sample_rate
        self.speech_threshold  = speech_threshold
        self._vad = SileroVAD()
        self._sed = PANNsSED(device=device)

    def segment_audio(self, audio: np.ndarray) -> List[SpeechSegment]:
        """
        Segments raw waveform using Silero-VAD into speech / non-speech intervals.
        PANNs CNN14 classifies each non-speech interval into a sound event.
        """
        frame_ms = SileroVAD.FRAME_MS
        probs    = self._vad.get_speech_probs(audio, self.sample_rate)

        if not probs:
            return []

        # Hangover smoothing — bridge short silences (< 300ms), suppress short speech (< 240ms)
        min_speech_frames  = max(1, int(240 / frame_ms))
        min_silence_frames = max(1, int(300 / frame_ms))

        flags = [p >= self.speech_threshold for p in probs]
        # Bridge short silence gaps
        gap_start = None
        for i, f in enumerate(flags):
            if not f:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None and (i - gap_start) < min_silence_frames:
                    for j in range(gap_start, i):
                        flags[j] = True
                gap_start = None

        # Cluster into contiguous segments
        segs: List[SpeechSegment] = []
        curr = flags[0]
        start_i = 0

        def _make_seg(start_f: int, end_f: int, is_sp: bool) -> SpeechSegment:
            s_sec = round(start_f * frame_ms / 1000.0, 3)
            e_sec = round(end_f   * frame_ms / 1000.0, 3)
            s_idx = int(s_sec * self.sample_rate)
            e_idx = int(e_sec * self.sample_rate)
            clip  = audio[s_idx:e_idx]
            rms   = float(np.sqrt(np.mean(clip ** 2) + 1e-12))
            rms_db = round(20 * np.log10(rms + 1e-12), 2)
            avg_p  = float(np.mean(probs[start_f:end_f]))

            sev = None
            if not is_sp:
                evt = self._sed.classify(clip, self.sample_rate) if len(clip) > 0 else "SILENCE"
                if evt is None:
                    evt = "MUSIC"
                lex = EVENT_LEXICON.get(evt, EVENT_LEXICON["MUSIC"])
                sev = SoundEvent(
                    event_type=evt,
                    start_sec=s_sec, end_sec=e_sec,
                    confidence=0.85,
                    bengali_tag=lex["bn"],
                    english_tag=lex["en"],
                    hindi_tag=lex["hi"],
                )
            return SpeechSegment(
                segment_id=len(segs) + 1,
                start_sec=s_sec, end_sec=e_sec,
                speech_prob=avg_p,
                energy_dbfs=rms_db,
                is_speech=is_sp,
                sound_event=sev,
            )

        for i in range(1, len(flags)):
            if flags[i] != curr:
                segs.append(_make_seg(start_i, i, curr))
                start_i = i
                curr = flags[i]
        segs.append(_make_seg(start_i, len(flags), curr))

        return segs

    def inspect_hallucination(
        self,
        cue_id: int,
        start_sec: float,
        end_sec: float,
        text: str,
        audio_slice: np.ndarray,
        whisper_no_speech_prob: float = 0.0,
        whisper_avg_logprob: float = 0.0,
    ) -> Optional[HallucinationAudit]:
        """
        Multi-signal Anti-Hallucination Guard.
        Targets the hackathon Toughest Test: phantom Whisper text over silence/music.

        Signals examined:
          1. Repetition loop  (zlib compression ratio)
          2. Acoustic energy  (RMS dBFS)
          3. Spectral flatness  (music vs speech)
          4. Silero-VAD speech probability over the slice
          5. Whisper internal no_speech_prob + avg_logprob
          6. Known phantom phrase patterns
        """
        if not text or not text.strip():
            return None

        clean = text.strip()
        reasons: List[str] = []
        risk = 0.0

        # 1. Repetition / compression loop
        compressed = zlib.compress(clean.encode("utf-8"))
        ratio = len(clean.encode("utf-8")) / (len(compressed) + 1e-6)
        word_repeat = any(clean.count(w) >= 4 for w in clean.split() if len(w) > 2)
        if ratio > 2.2 or word_repeat:
            reasons.append(f"REPETITION_HALLUCINATION (ratio={ratio:.2f})")
            risk += 0.55

        # 2 & 3. Acoustic energy + spectral flatness
        if len(audio_slice) > 0:
            rms    = np.sqrt(np.mean(audio_slice ** 2) + 1e-12)
            rms_db = 20 * np.log10(rms + 1e-12)
            fft    = np.abs(np.fft.rfft(audio_slice[:min(len(audio_slice), 32000)])) + 1e-12
            flat   = float(np.exp(np.mean(np.log(fft))) / (np.mean(fft) + 1e-12))

            if rms_db < -46.0:
                reasons.append(f"HALLUCINATION_OVER_SILENCE (RMS={rms_db:.1f} dBFS)")
                risk += 0.65
            elif rms_db < -38.0 and flat < 0.22:
                reasons.append(f"HALLUCINATION_OVER_MUSIC (flatness={flat:.3f})")
                risk += 0.50

        # 4. Silero-VAD speech probability over slice
        if len(audio_slice) > int(self.sample_rate * 0.3):
            vad_probs = self._vad.get_speech_probs(audio_slice, self.sample_rate)
            if vad_probs:
                avg_vad = float(np.mean(vad_probs))
                if avg_vad < 0.20:
                    reasons.append(f"LOW_VAD_SPEECH_PROB ({avg_vad:.2f})")
                    risk += 0.45

        # 5. Whisper internal flags
        if whisper_no_speech_prob > 0.55:
            reasons.append(f"WHISPER_NO_SPEECH ({whisper_no_speech_prob:.2f})")
            risk += 0.45
        if whisper_avg_logprob < -1.15:
            reasons.append(f"LOW_TOKEN_LOGPROB ({whisper_avg_logprob:.2f})")
            risk += 0.35

        # 6. Known phantom phrases
        PHANTOMS = [
            "subscribe", "thank you", "thanks for watching",
            "লাইক করুন", "শেয়ার করুন", "সাবস্ক্রাইব",
            "please", "bye", "you",
        ]
        if any(p in clean.lower() for p in PHANTOMS) and (whisper_no_speech_prob > 0.35 or risk > 0.2):
            reasons.append("KNOWN_PHANTOM_PHRASE")
            risk += 0.50

        risk = min(1.0, risk)

        if risk >= 0.40:
            action      = "SUPPRESS" if risk >= 0.70 else "FLAG_FOR_HUMAN_REVIEW"
            replacement = "[MUSIC PLAYING]" if any("MUSIC" in r for r in reasons) else "[SILENCE]"
            return HallucinationAudit(
                cue_id=cue_id,
                start_sec=start_sec, end_sec=end_sec,
                hallucinated_text=clean,
                risk_score=round(risk, 3),
                risk_reasons=reasons,
                recommended_action=action,
                replacement_event=replacement,
            )

        return None
