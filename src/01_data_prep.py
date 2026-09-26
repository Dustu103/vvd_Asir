"""
01_data_prep.py  —  MUCS 2021 Bengali-English code-switching data prep
=======================================================================

Dataset: MUCS 2021 Indic ASR Challenge — Bengali-English track
  Domain  : Computer Science tech lectures (16 kHz)
  Train   : 46.11 hours  →  OpenSLR resource 104
  Test    :  7.02 hours  →  OpenSLR resource 104
  License : CC BY-SA 4.0
  Note    : Heavy CS terminology mid-sentence (algorithm, CPU, pointer, etc.)
            These English tech terms are THE critical words for EWER scoring.

Download URLs (direct, no registration):
  Train : https://www.openslr.org/resources/104/Bengali-English_train.tar.gz  (~3.9 GB)
  Test  : https://www.openslr.org/resources/104/Bengali-English_test.tar.gz   (~606 MB)

This script will auto-download if --data_root is empty.
Or pass --skip_download if you already extracted the data manually.

What this script does:
  1. Mounts Google Drive (Colab) or uses a local path
  2. Downloads + extracts train split if not already present
  3. Detects utterances with ≥1 Bengali↔Latin script switch
  4. Splits into 3 tiers by switch density (high / mid / base)
  5. Stratified-samples 1500–2500 utterances, oversamples high-switch tiers
  6. Saves per-tier CSVs + master manifest to --out_dir
"""

import os, sys, re, csv, json, random, argparse, unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Unicode block helpers ────────────────────────────────────────────────────
BENGALI_RANGE = (0x0980, 0x09FF)   # Bengali Unicode block
LATIN_RANGE   = (0x0041, 0x007A)   # Basic Latin A-z (covers English)

def char_script(ch: str) -> str:
    """Return 'bengali', 'latin', or 'other' for a single character."""
    cp = ord(ch)
    if BENGALI_RANGE[0] <= cp <= BENGALI_RANGE[1]:
        return "bengali"
    if LATIN_RANGE[0] <= cp <= LATIN_RANGE[1]:
        return "latin"
    return "other"

def count_script_switches(text: str) -> int:
    """
    Count the number of Bengali↔Latin boundary crossings in the transcript.
    e.g. 'আমি go করব' → 2 switches (বাংলা→latin, latin→বাংলা)
    """
    scripts = [char_script(c) for c in text if char_script(c) != "other"]
    if not scripts:
        return 0
    switches = 0
    prev = scripts[0]
    for s in scripts[1:]:
        if s != prev:
            switches += 1
            prev = s
    return switches

def has_both_scripts(text: str) -> bool:
    scripts = {char_script(c) for c in text if char_script(c) != "other"}
    return "bengali" in scripts and "latin" in scripts

# ── Tier assignment ──────────────────────────────────────────────────────────
def assign_tier(n_switches: int) -> str:
    if n_switches >= 3:
        return "high"      # Tier A — dense switches, oversample 3×
    elif n_switches == 2:
        return "mid"       # Tier B — moderate switches, oversample 2×
    else:
        return "base"      # Tier C — single switch / low density, 1×

OVERSAMPLE = {"high": 3, "mid": 2, "base": 1}

# ── MUCS manifest parser ─────────────────────────────────────────────────────
# ── Download URLs ───────────────────────────────────────────────────────────────
TRAIN_URL = "https://www.openslr.org/resources/104/Bengali-English_train.tar.gz"
TEST_URL  = "https://www.openslr.org/resources/104/Bengali-English_test.tar.gz"


def download_and_extract(data_root: Path, skip_download: bool = False) -> None:
    """
    Download MUCS Bengali-English train split and extract it.
    Safe to call if data is already present — skips automatically.
    """
    import tarfile, zipfile, urllib.request

    data_root.mkdir(parents=True, exist_ok=True)
    already_extracted = bool(list(data_root.rglob("transcription*.txt")))

    if already_extracted:
        print("  ✅ Data already extracted — skipping download.")
        return

    if skip_download:
        print("  ⚠️  --skip_download set but no transcription.txt found.")
        print(f"  Place extracted data under: {data_root}")
        return

    # ── Download train (tar.gz, ~2 GB) ───────────────────────────────────────
    tar_path = data_root / "Bengali-English_train.tar.gz"
    if not tar_path.exists():
        print(f"  Downloading train split (~2 GB) …")
        print(f"  URL: {TRAIN_URL}")

        def _reporthook(count, block_size, total_size):
            done = count * block_size
            pct  = min(100, int(done * 100 / max(total_size, 1)))
            mb   = done / 1e6
            print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(TRAIN_URL, tar_path, reporthook=_reporthook)
        print()  # newline after progress
        print(f"  Downloaded → {tar_path}")
    else:
        print(f"  Archive already downloaded: {tar_path}")

    # ── Extract ───────────────────────────────────────────────────────────────
    print("  Extracting …")
    if str(tar_path).endswith(".tar.gz") or str(tar_path).endswith(".tgz"):
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(data_root)
    else:
        with zipfile.ZipFile(tar_path, "r") as zf:
            zf.extractall(data_root)
    print(f"  Extracted → {data_root}")


def find_mucs_files(data_root: Path):
    """
    Locate transcript file and audio directory inside extracted MUCS data.
    Supports official OpenSLR Kaldi layout (transcripts/text + segments) as well
    as flat transcription.txt layouts.
    """
    data_root = Path(data_root)

    # 1. Check for official OpenSLR Kaldi format
    kaldi_text = [p for p in data_root.rglob("text") if "test" not in str(p).lower() and p.is_file()]
    kaldi_segs = [p for p in data_root.rglob("segments") if "test" not in str(p).lower() and p.is_file()]
    if kaldi_text and kaldi_segs:
        return ("kaldi", kaldi_text[0], kaldi_segs[0], data_root)

    # 2. Fallback to flat layout
    transcript_candidates = (
        list(data_root.rglob("transcription*.txt")) +
        list(data_root.rglob("transcript*.txt"))    +
        list(data_root.rglob("*.tsv"))
    )
    transcript_candidates = [
        p for p in transcript_candidates if "test" not in str(p).lower()
    ]
    audio_candidates = list(data_root.rglob("*.wav"))

    if not transcript_candidates:
        return None, None, None, None

    transcript_path = transcript_candidates[0]
    audio_dir = audio_candidates[0].parent if audio_candidates else None
    return ("flat", transcript_path, None, audio_dir)


def parse_mucs_transcript(format_type: str, transcript_path: Path, segments_path: Path, data_dir: Path):
    """
    Parse MUCS transcript into a list of dicts:
      { utt_id, audio_path, start_sec, end_sec, duration, transcript, n_switches, tier }
    """
    rows = []
    if format_type == "kaldi":
        # Parse Kaldi segments: utt_id -> (rec_id, start_sec, end_sec)
        wav_map = {p.stem: str(p) for p in data_dir.rglob("*.wav")}
        seg_dict = {}
        with open(segments_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    seg_dict[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))

        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                uid, raw_text = parts[0], parts[1].strip()
                if uid not in seg_dict:
                    continue
                rec_id, s_sec, e_sec = seg_dict[uid]
                apath = wav_map.get(rec_id)
                if not apath:
                    continue

                trans = unicodedata.normalize("NFC", raw_text)
                n_sw = count_script_switches(trans)
                rows.append({
                    "utt_id":     uid,
                    "audio_path": apath,
                    "start_sec":  s_sec,
                    "end_sec":    e_sec,
                    "duration":   round(e_sec - s_sec, 2),
                    "transcript": trans,
                    "n_switches": n_sw,
                    "tier":       assign_tier(n_sw),
                })
        return rows

    # Flat layout
    audio_dir = data_dir
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) < 2:
                continue
            utt_id, transcript = parts[0].strip(), parts[1].strip()

            audio_path = None
            if audio_dir:
                for ext in [".wav", ".flac", ".mp3"]:
                    candidate = audio_dir / (utt_id + ext)
                    if candidate.exists():
                        audio_path = str(candidate)
                        break

            n_sw = count_script_switches(transcript)
            rows.append({
                "utt_id":     utt_id,
                "audio_path": audio_path,
                "start_sec":  0.0,
                "end_sec":    30.0,
                "duration":   30.0,
                "transcript": transcript,
                "n_switches": n_sw,
                "tier":       assign_tier(n_sw),
            })
    return rows

# ── Stratified sampling with oversampling ────────────────────────────────────
def build_training_set(rows: list, target_total: int = 2000, seed: int = 42):
    """
    1. Keep only utterances with ≥1 script switch.
    2. Oversample high/mid tiers.
    3. Stratified-sample to target_total.
    Returns train_rows, val_rows (90/10 split).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter: must have at least one switch
    switched = [r for r in rows if r["n_switches"] >= 1]
    print(f"  Utterances with ≥1 script switch : {len(switched)}")

    if not switched:
        raise ValueError(
            "No code-switched utterances found. "
            "Check your data_root — make sure the Bengali-English split is there."
        )

    # Oversample
    expanded = []
    for r in switched:
        factor = OVERSAMPLE[r["tier"]]
        expanded.extend([r] * factor)

    # Shuffle
    random.shuffle(expanded)

    # Cap at target_total (stratified by tier)
    df = pd.DataFrame(expanded)
    tier_counts = df["tier"].value_counts(normalize=True)
    sampled_dfs = []
    for tier, frac in tier_counts.items():
        n = max(1, round(frac * target_total))
        tier_df = df[df["tier"] == tier]
        n = min(n, len(tier_df))
        sampled_dfs.append(tier_df.sample(n, random_state=seed))
    sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  After oversampling + sampling    : {len(sampled)} utterances")
    print(f"  Tier breakdown:\n{sampled['tier'].value_counts().to_string()}")

    # Train / val split (90/10), stratified by tier
    train_df, val_df = train_test_split(
        sampled, test_size=0.10, random_state=seed, stratify=sampled["tier"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ── Colab Drive mount ────────────────────────────────────────────────────────
def try_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted at /content/drive")
    except Exception:
        print("ℹ️  Not running in Colab — skipping Drive mount.")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MUCS data prep for Bengali-English CS ASR")
    parser.add_argument("--data_root",    default="/content/drive/MyDrive/hoichoi/mucs_raw",
                        help="Root directory of extracted MUCS data")
    parser.add_argument("--out_dir",      default="/content/drive/MyDrive/hoichoi/data",
                        help="Where to save manifests (CSVs)")
    parser.add_argument("--target_total",  type=int, default=2000,
                        help="Target training set size (1500-2500 recommended)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download — assume data is already extracted at --data_root")
    parser.add_argument("--no_drive",      action="store_true",
                        help="Skip Google Drive mount (local run)")
    args = parser.parse_args()

    if not args.no_drive:
        try_mount_drive()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Locate MUCS files ────────────────────────────────────────────
    print(f"\n[1/4] Scanning {data_root} for MUCS files …")
    # ── Download + extract (skips automatically if already done) ────────────
    download_and_extract(data_root, skip_download=args.skip_download)

    fmt_type, transcript_path, segments_path, audio_or_data_dir = find_mucs_files(data_root)
    if not fmt_type:
        print(
            f"\n❌ No transcript data found under: {data_root}\n"
            "   If download failed, try downloading manually:\n"
            f"   wget '{TRAIN_URL}' -O Bengali-English_train.tar.gz\n"
            "   tar -xzf Bengali-English_train.tar.gz\n"
        )
        sys.exit(1)

    print(f"  Format detected : {fmt_type.upper()}")
    print(f"  Transcript file : {transcript_path}")
    if segments_path:
        print(f"  Segments file   : {segments_path}")

    # ── Step 2: Parse & count switches ──────────────────────────────────────
    print("\n[2/4] Parsing transcripts and counting script switches …")
    rows = parse_mucs_transcript(fmt_type, transcript_path, segments_path, audio_or_data_dir)
    print(f"  Total utterances parsed          : {len(rows)}")

    # Save full stats for inspection
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "all_utterances_stats.csv", index=False, encoding="utf-8")
    print(f"  Full stats saved → {out_dir / 'all_utterances_stats.csv'}")

    # ── Step 3: Sample & tier-split ──────────────────────────────────────────
    print(f"\n[3/4] Stratified sampling (target={args.target_total}) …")
    train_df, val_df = build_training_set(rows, target_total=args.target_total, seed=args.seed)

    # ── Step 4: Save per-tier manifests + master ─────────────────────────────
    print(f"\n[4/4] Saving manifests to {out_dir} …")

    train_df.to_csv(out_dir / "train_all.csv",   index=False, encoding="utf-8")
    val_df.to_csv(  out_dir / "val.csv",          index=False, encoding="utf-8")

    for tier in ["high", "mid", "base"]:
        tier_df = train_df[train_df["tier"] == tier]
        tier_df.to_csv(out_dir / f"train_{tier}.csv", index=False, encoding="utf-8")
        print(f"  train_{tier}.csv  → {len(tier_df)} rows")

    # Save config snapshot
    config = {
        "data_root":    str(data_root),
        "transcript":   str(transcript_path),
        "target_total": args.target_total,
        "seed":         args.seed,
        "train_size":   len(train_df),
        "val_size":     len(val_df),
        "tiers":        {t: int((train_df["tier"] == t).sum()) for t in ["high", "mid", "base"]},
    }
    (out_dir / "data_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n✅ Done!")
    print(f"   Train : {len(train_df)}  |  Val : {len(val_df)}")
    print(f"   Next  : run 02_train_adapter.py --tier high (then mid, then base)")

if __name__ == "__main__":
    main()
