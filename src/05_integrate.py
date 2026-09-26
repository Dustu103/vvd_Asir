"""
05_integrate.py  —  Drop-in replacement for the Gated ASR box
=============================================================
Same input/output contract as the splice-heuristic ASR:
  Input  : VAD-gated, diarized audio segment (numpy array or file path)
  Output : { "text": str, "timestamps": [...], "language": "bn-en" }

Only swap in if 04_evaluate.py gives a clear win.
Usage:
  from src.05_integrate import GatedASR
  asr = GatedASR(adapter_path="/content/drive/MyDrive/hoichoi/adapters/merged")
  result = asr.transcribe("path/to/clip.wav")
"""

import os, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import torch


@dataclass
class ASRResult:
    text:       str
    timestamps: List[dict]      # [{"word": str, "start": float, "end": float}]
    language:   str = "bn-en"
    confidence: Optional[float] = None


class GatedASR:
    """
    Drop-in replacement for the splice-heuristic Gated ASR box.

    Example
    -------
    asr = GatedASR(
        adapter_path = "/content/drive/MyDrive/hoichoi/adapters/merged",
        base_model   = "SayedShaun/bengali-whisper-medium",
        device       = "cuda",
    )
    result = asr.transcribe("segment.wav")
    print(result.text)           # NFC-normalized mixed-script transcript
    print(result.timestamps)     # word-level timestamps
    """

    def __init__(
        self,
        adapter_path: str,
        base_model:   str  = "SayedShaun/bengali-whisper-medium",
        device:       str  = "auto",
        sample_rate:  int  = 16_000,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device      = device
        self.sample_rate = sample_rate

        self._load_model(adapter_path, base_model)

    def _load_model(self, adapter_path: str, base_model: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel

        print(f"[GatedASR] Loading base model: {base_model}")
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        print(f"[GatedASR] Attaching LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model = self.model.to(self.device).eval()

        self.processor = WhisperProcessor.from_pretrained(adapter_path)
        print("[GatedASR] Ready ✅")

    # ── Public API (same contract as splice-heuristic baseline) ──────────────
    def transcribe(
        self,
        audio: Union[str, np.ndarray, "torch.Tensor"],
        return_timestamps: bool = True,
    ) -> ASRResult:
        """
        Parameters
        ----------
        audio : str | np.ndarray | torch.Tensor
            File path OR raw waveform at self.sample_rate Hz (mono).
        return_timestamps : bool
            If True, returns word-level timestamps (Whisper's built-in).

        Returns
        -------
        ASRResult with .text (NFC-normalized) and .timestamps
        """
        waveform = self._load_audio(audio)
        return self._run_inference(waveform, return_timestamps)

    # ── Internals ─────────────────────────────────────────────────────────────
    def _load_audio(self, audio) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            import librosa
            wav, _ = librosa.load(str(audio), sr=self.sample_rate, mono=True)
            return wav
        elif isinstance(audio, torch.Tensor):
            return audio.cpu().numpy()
        elif isinstance(audio, np.ndarray):
            return audio
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _run_inference(self, waveform: np.ndarray, return_timestamps: bool) -> ASRResult:
        # Truncate to Whisper max (30 s)
        max_samples = 30 * self.sample_rate
        waveform    = waveform[:max_samples]

        inputs = self.processor(
            waveform,
            sampling_rate = self.sample_rate,
            return_tensors = "pt",
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        generate_kwargs = dict(
            language       = "Bengali",
            task           = "transcribe",
            max_new_tokens = 448,
        )
        if return_timestamps:
            generate_kwargs["return_timestamps"] = True

        with torch.no_grad():
            outputs = self.model.generate(input_features, **generate_kwargs)

        # Decode
        if return_timestamps:
            decoded = self.processor.batch_decode(outputs, output_offsets=True)
            raw_text   = decoded[0]["text"] if decoded else ""
            timestamps = decoded[0].get("offsets", [])
            # Normalise timestamp format → list of {"word", "start", "end"}
            ts_out = [
                {
                    "word":  chunk.get("text", "").strip(),
                    "start": chunk.get("timestamp", [None, None])[0],
                    "end":   chunk.get("timestamp", [None, None])[1],
                }
                for chunk in timestamps
            ]
        else:
            raw_text   = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            ts_out     = []

        # NFC normalize (mandatory — per spec)
        text = unicodedata.normalize("NFC", raw_text).strip()

        return ASRResult(text=text, timestamps=ts_out)

    # ── Batch variant (for pipeline throughput) ───────────────────────────────
    def transcribe_batch(
        self,
        audio_list: List[Union[str, np.ndarray]],
        return_timestamps: bool = False,
    ) -> List[ASRResult]:
        return [self.transcribe(a, return_timestamps) for a in audio_list]


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",   required=True)
    parser.add_argument("--audio",     required=True, help="Path to .wav file")
    parser.add_argument("--base_model",default="SayedShaun/bengali-whisper-medium")
    args = parser.parse_args()

    asr    = GatedASR(adapter_path=args.adapter, base_model=args.base_model)
    result = asr.transcribe(args.audio, return_timestamps=True)

    print("\n── Transcript ──────────────────────────────────────")
    print(result.text)
    print("\n── Timestamps ──────────────────────────────────────")
    print(json.dumps(result.timestamps, ensure_ascii=False, indent=2))
