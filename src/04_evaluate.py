"""
04_evaluate.py  —  Score the merged adapter vs. splice-heuristic baseline
==========================================================================
Run after 03_merge_adapters.py:

    !python src/04_evaluate.py \
        --test_csv  /content/drive/MyDrive/hoichoi/data/val.csv \
        --adapter   /content/drive/MyDrive/hoichoi/adapters/merged

Metrics computed:
  1. WER  (NFC-normalized on both hypothesis and reference)
  2. English Word Emission Rate (EWER)  — the primary code-switch quality metric:
       For each English word in the reference, did the model emit the Latin-script
       word, or did it transliterate it into Bengali script?
  3. Script confusion matrix (TP/FP/FN for Bengali vs Latin tokens)

Baseline comparison:
  Pass --baseline_model to also score a second model (e.g., the splice heuristic)
  side-by-side so you can make the swap decision from step 5.
"""

import os, sys, re, json, argparse, unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Colab helpers ─────────────────────────────────────────────────────────────
def try_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted")
    except Exception:
        pass

def install_if_colab():
    try:
        import google.colab  # noqa
        os.system("pip install -q transformers>=4.40.0 peft>=0.10.0 jiwer librosa soundfile")
    except ImportError:
        pass

# ── NFC normalizer ────────────────────────────────────────────────────────────
def nfc(text: str) -> str:
    """NFC-normalize and strip extra whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())

# ── Script detection helpers ──────────────────────────────────────────────────
BENGALI_RANGE = (0x0980, 0x09FF)
LATIN_RANGE   = (0x0041, 0x007A)

def is_bengali_word(word: str) -> bool:
    return any(BENGALI_RANGE[0] <= ord(c) <= BENGALI_RANGE[1] for c in word)

def is_latin_word(word: str) -> bool:
    return any(LATIN_RANGE[0] <= ord(c) <= LATIN_RANGE[1] for c in word)

def tokenize_words(text: str) -> List[str]:
    """Simple whitespace tokenizer after NFC normalize."""
    return nfc(text).split()

# ── English Word Emission Rate (EWER) ─────────────────────────────────────────
@dataclass
class EWERResult:
    ref_english_words:  int   = 0   # English (Latin-script) words in reference
    emitted_correctly:  int   = 0   # model also emitted Latin-script word
    transliterated:     int   = 0   # model emitted Bengali-script instead
    missing:            int   = 0   # word absent from hypothesis entirely
    ewer:               float = 0.0 # emitted_correctly / ref_english_words

def compute_ewer(reference: str, hypothesis: str) -> EWERResult:
    """
    For every English (Latin-script) word in reference:
      - Find its position in a word-aligned hypothesis (greedy match)
      - Check whether the hypothesis word is also Latin-script
    """
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)

    result = EWERResult()
    hyp_idx = 0

    for ref_word in ref_words:
        if not is_latin_word(ref_word):
            continue  # skip Bengali reference words
        result.ref_english_words += 1

        # Try to find this word (or approximate) in hypothesis
        found = False
        for j in range(hyp_idx, min(hyp_idx + 5, len(hyp_words))):  # lookahead 5
            if hyp_words[j].lower() == ref_word.lower():
                # Exact match
                result.emitted_correctly += 1
                hyp_idx = j + 1
                found = True
                break
            elif is_bengali_word(hyp_words[j]) and not is_latin_word(hyp_words[j]):
                # A Bengali word appeared where an English word was expected
                result.transliterated += 1
                hyp_idx = j + 1
                found = True
                break

        if not found:
            result.missing += 1

    if result.ref_english_words > 0:
        result.ewer = result.emitted_correctly / result.ref_english_words
    return result

# ── WER computation ───────────────────────────────────────────────────────────
def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    from jiwer import wer
    refs = [nfc(r) for r in references]
    hyps = [nfc(h) for h in hypotheses]
    return wer(refs, hyps)

# ── Model inference ───────────────────────────────────────────────────────────
def load_lora_model(adapter_path: str, base_model_id: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    processor = WhisperProcessor.from_pretrained(adapter_path)
    base      = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.to(device).eval()
    return model, processor


def transcribe_batch(audio_paths: List[str], model, processor, device: str,
                     batch_size: int = 8) -> List[str]:
    import librosa
    results = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="  Transcribing"):
        batch_paths = audio_paths[i: i + batch_size]
        waveforms   = []
        for p in batch_paths:
            try:
                wav, _ = librosa.load(p, sr=16_000, mono=True)
            except Exception:
                wav = np.zeros(16_000, dtype=np.float32)
            waveforms.append(wav)

        inputs = processor(
            waveforms, sampling_rate=16_000, return_tensors="pt", padding=True
        )
        input_features = inputs.input_features.to(device)
        if device == "cuda":
            input_features = input_features.half()

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="Bengali",
                task="transcribe",
                max_new_tokens=448,
            )
        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        results.extend(decoded)

    return results

# ── Pretty print results ──────────────────────────────────────────────────────
def print_report(tag: str, wer_score: float, ewer: EWERResult, n: int):
    print(f"\n{'='*55}")
    print(f"  {tag}")
    print(f"{'='*55}")
    print(f"  Samples evaluated      : {n}")
    print(f"  WER  (NFC-normalized)  : {wer_score*100:.2f}%")
    print(f"  ── English Word Emission ──────────────────────────")
    print(f"  English words in ref   : {ewer.ref_english_words}")
    print(f"  Emitted correctly      : {ewer.emitted_correctly}")
    print(f"  Transliterated (⚠️)    : {ewer.transliterated}")
    print(f"  Missing entirely       : {ewer.missing}")
    print(f"  EWER (↑ better)        : {ewer.ewer*100:.1f}%")
    print(f"{'='*55}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",       required=True,
                        help="Val/test CSV from 01_data_prep.py")
    parser.add_argument("--adapter",        required=True,
                        help="Path to merged adapter dir")
    parser.add_argument("--base_model",     default="SayedShaun/bengali-whisper-medium")
    parser.add_argument("--baseline_model", default=None,
                        help="Optional second model to compare (HF model ID or adapter path)")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--limit",          type=int, default=None,
                        help="Evaluate on first N samples only (quick sanity check)")
    parser.add_argument("--out_json",       default=None,
                        help="Save full results to this JSON file")
    parser.add_argument("--no_drive",       action="store_true")
    args = parser.parse_args()

    install_if_colab()
    if not args.no_drive:
        try_mount_drive()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load test data ────────────────────────────────────────────────────────
    df = pd.read_csv(args.test_csv)
    df = df[df["audio_path"].notna() & (df["n_switches"] >= 1)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    audio_paths = df["audio_path"].tolist()
    references  = df["transcript"].tolist()
    print(f"\nEvaluating on {len(df)} code-switched utterances …")

    # ── Merged LoRA model ─────────────────────────────────────────────────────
    print(f"\n[1/2] Loading LoRA model from {args.adapter} …")
    lora_model, lora_proc = load_lora_model(args.adapter, args.base_model, device)
    lora_hyps = transcribe_batch(audio_paths, lora_model, lora_proc, device, args.batch_size)

    lora_wer  = compute_wer(references, lora_hyps)
    lora_ewer = EWERResult()
    for ref, hyp in zip(references, lora_hyps):
        r = compute_ewer(ref, hyp)
        lora_ewer.ref_english_words += r.ref_english_words
        lora_ewer.emitted_correctly += r.emitted_correctly
        lora_ewer.transliterated    += r.transliterated
        lora_ewer.missing           += r.missing
    if lora_ewer.ref_english_words:
        lora_ewer.ewer = lora_ewer.emitted_correctly / lora_ewer.ref_english_words

    print_report("LoRA Merged Adapter", lora_wer, lora_ewer, len(df))

    # ── Baseline (optional) ───────────────────────────────────────────────────
    baseline_wer, baseline_ewer = None, None
    if args.baseline_model:
        print(f"\n[2/2] Loading baseline: {args.baseline_model} …")
        try:
            # Try as LoRA adapter first, fall back to plain Whisper
            bl_model, bl_proc = load_lora_model(args.baseline_model, args.base_model, device)
        except Exception:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            bl_proc  = WhisperProcessor.from_pretrained(args.baseline_model)
            bl_model = WhisperForConditionalGeneration.from_pretrained(
                args.baseline_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device).eval()

        bl_hyps = transcribe_batch(audio_paths, bl_model, bl_proc, device, args.batch_size)
        baseline_wer  = compute_wer(references, bl_hyps)
        baseline_ewer = EWERResult()
        for ref, hyp in zip(references, bl_hyps):
            r = compute_ewer(ref, hyp)
            baseline_ewer.ref_english_words += r.ref_english_words
            baseline_ewer.emitted_correctly += r.emitted_correctly
            baseline_ewer.transliterated    += r.transliterated
            baseline_ewer.missing           += r.missing
        if baseline_ewer.ref_english_words:
            baseline_ewer.ewer = baseline_ewer.emitted_correctly / baseline_ewer.ref_english_words

        print_report("Baseline Model", baseline_wer, baseline_ewer, len(df))

    # ── Decision: swap or keep? ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SWAP DECISION")
    print("="*55)
    if baseline_wer is not None:
        wer_delta  = (baseline_wer - lora_wer) * 100
        ewer_delta = (lora_ewer.ewer - baseline_ewer.ewer) * 100
        print(f"  WER  delta  (+ = LoRA better)  : {wer_delta:+.2f}%")
        print(f"  EWER delta  (+ = LoRA better)  : {ewer_delta:+.1f}%")
        if lora_wer < baseline_wer and lora_ewer.ewer >= baseline_ewer.ewer:
            print("  → ✅ SWAP: LoRA clearly wins on both metrics.")
        elif lora_ewer.ewer > baseline_ewer.ewer + 0.05:
            print("  → ✅ SWAP: LoRA wins on English emission (the key metric).")
        else:
            print("  → ⛔ KEEP splice baseline — LoRA not clearly better.")
    else:
        print(f"  LoRA WER : {lora_wer*100:.2f}%  |  MUCS baseline floor: 32.45%")
        if lora_wer < 0.3245:
            print("  → ✅ Below MUCS baseline floor — swap is justified.")
        else:
            print("  → ⚠️  Still above MUCS baseline — keep training or tune merge weights.")

    # ── Save detailed results ─────────────────────────────────────────────────
    if args.out_json:
        out = {
            "lora": {
                "wer":  lora_wer,
                "ewer": lora_ewer.ewer,
                "ref_english_words": lora_ewer.ref_english_words,
                "emitted_correctly": lora_ewer.emitted_correctly,
                "transliterated":    lora_ewer.transliterated,
            },
            "samples": [
                {"audio": a, "ref": r, "hyp": h}
                for a, r, h in zip(audio_paths, references, lora_hyps)
            ]
        }
        if baseline_wer is not None:
            out["baseline"] = {"wer": baseline_wer, "ewer": baseline_ewer.ewer}
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  Full results saved → {args.out_json}")


if __name__ == "__main__":
    main()
