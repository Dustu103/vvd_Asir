"""
02_train_adapter.py  —  Train a single LoRA adapter for one data tier
======================================================================
Run 3 times in separate Colab sessions (or sequentially):

    !python src/02_train_adapter.py --tier high
    !python src/02_train_adapter.py --tier mid
    !python src/02_train_adapter.py --tier base

Each session saves its adapter to:
    /content/drive/MyDrive/hoichoi/adapters/{tier}/

Colab T4 (16 GB) settings are the default. Tune --batch_size and
--grad_accum if you get OOM (reduce batch) or too slow (increase accum).
"""

import os, sys, json, argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Colab Drive mount ────────────────────────────────────────────────────────
def try_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted")
    except Exception:
        print("ℹ️  Not in Colab — skipping Drive mount.")

# ── Installs (Colab only) ────────────────────────────────────────────────────
def install_if_colab():
    try:
        import google.colab  # noqa
        os.system(
            "pip install -q transformers>=4.40.0 peft>=0.10.0 "
            "datasets>=2.18.0 accelerate>=0.29.0 "
            "librosa soundfile jiwer"
        )
    except ImportError:
        pass

# ── Dataset ──────────────────────────────────────────────────────────────────
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

class MUCSDataset(Dataset):
    """
    Reads a manifest CSV (utt_id, audio_path, transcript, …)
    Returns dict ready for Whisper's feature extractor.
    """
    def __init__(self, csv_path: str, feature_extractor, tokenizer,
                 sample_rate: int = 16_000, max_duration_s: float = 30.0):
        self.df = pd.read_csv(csv_path)
        # Drop rows with missing audio
        self.df = self.df[self.df["audio_path"].notna()].reset_index(drop=True)
        self.fe        = feature_extractor
        self.tok       = tokenizer
        self.sr        = sample_rate
        self.max_len   = int(max_duration_s * sample_rate)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s_sec = float(row.get("start_sec", 0.0))
        e_sec = float(row.get("end_sec", 0.0))
        dur   = (e_sec - s_sec) if e_sec > s_sec else None
        # Load audio
        try:
            audio, sr = librosa.load(row["audio_path"], sr=self.sr, mono=True,
                                     offset=s_sec, duration=min(30.0, dur) if dur else 30.0)
        except Exception as e:
            print(f"⚠️  Could not load {row['audio_path']}: {e}")
            audio = np.zeros(self.sr, dtype=np.float32)  # 1s silence fallback

        # Truncate to Whisper max (30s)
        audio = audio[: self.max_len]

        # Feature extraction (log-mel spectrogram)
        inputs = self.fe(audio, sampling_rate=self.sr, return_tensors="pt")
        input_features = inputs.input_features.squeeze(0)   # [80, 3000]

        # Tokenize transcript (NFC normalize first)
        import unicodedata
        transcript = unicodedata.normalize("NFC", str(row["transcript"]))
        labels = self.tok(transcript, return_tensors="pt").input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch, tokenizer):
    """Pad input features and labels to batch max length."""
    from transformers import WhisperProcessor
    input_features = [b["input_features"] for b in batch]
    labels         = [b["labels"]          for b in batch]

    # Whisper feature extractor already pads to 3000 frames; stack directly
    input_features = torch.stack(input_features)  # [B, 80, 3000]

    # Pad labels with -100 (ignore index)
    max_len = max(l.size(0) for l in labels)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :l.size(0)] = l

    return {"input_features": input_features, "labels": padded_labels}


# ── LoRA config ──────────────────────────────────────────────────────────────
def build_lora_model(base_model_id: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  Loading base model: {base_model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    # Enable gradient checkpointing to save VRAM
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r             = 32,
        lora_alpha    = 64,
        target_modules= ["q_proj", "v_proj"],   # Tanglish-proven config
        lora_dropout  = 0.05,
        bias          = "none",
        task_type     = TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training tier='{args.tier}' on device={device}")

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    adapter_dir = Path(args.adapter_dir) / args.tier
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / f"train_{args.tier}.csv"
    val_csv   = data_dir / "val.csv"

    if not train_csv.exists():
        print(f"❌ {train_csv} not found. Run 01_data_prep.py first.")
        sys.exit(1)

    # ── Model & processor ───────────────────────────────────────────────────
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.base_model, language="Bengali", task="transcribe")
    feature_extractor = processor.feature_extractor
    tokenizer         = processor.tokenizer

    model = build_lora_model(args.base_model, device)

    # ── Data ────────────────────────────────────────────────────────────────
    from functools import partial
    train_ds = MUCSDataset(train_csv, feature_extractor, tokenizer)
    val_ds   = MUCSDataset(val_csv,   feature_extractor, tokenizer)

    collate = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=(device == "cuda")
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ── Checkpoint every 20% of planned steps ───────────────────────────────
    total_steps       = args.max_steps
    checkpoint_every  = max(1, total_steps // 5)
    best_val_loss     = float("inf")
    plateau_patience  = 5
    no_improve_count  = 0
    global_step       = 0

    print(f"  Total steps      : {total_steps}")
    print(f"  Checkpoint every : {checkpoint_every} steps")
    print(f"  Train samples    : {len(train_ds)}")
    print(f"  Val samples      : {len(val_ds)}\n")

    model.train()

    def run_val():
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_features"].to(device)
                lbl = batch["labels"].to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    out = model(input_features=inp, labels=lbl)
                total_loss += out.loss.item() * inp.size(0)
                n += inp.size(0)
        model.train()
        return total_loss / max(n, 1)

    grad_accum_step = 0

    for epoch in range(999):  # infinite epochs — stop on plateau
        for batch in train_loader:
            if global_step >= total_steps:
                break

            inp = batch["input_features"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out   = model(input_features=inp, labels=lbl)
                loss  = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            grad_accum_step += 1

            if grad_accum_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                grad_accum_step = 0

                # ── Logging ──────────────────────────────────────────────────
                if global_step % 10 == 0:
                    print(f"  step {global_step:>5}/{total_steps}  train_loss={loss.item()*args.grad_accum:.4f}")

                # ── Checkpoint ───────────────────────────────────────────────
                if global_step % checkpoint_every == 0 or global_step == total_steps:
                    val_loss = run_val()
                    print(f"  ✦ step {global_step}  val_loss={val_loss:.4f}")

                    ckpt_path = adapter_dir / f"checkpoint_step{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    processor.save_pretrained(str(ckpt_path))
                    print(f"    Saved checkpoint → {ckpt_path}")

                    # Best model
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss    = val_loss
                        no_improve_count = 0
                        best_path = adapter_dir / "best"
                        model.save_pretrained(str(best_path))
                        processor.save_pretrained(str(best_path))
                        print(f"    ✅ New best val_loss={best_val_loss:.4f} → {best_path}")
                    else:
                        no_improve_count += 1
                        print(f"    No improvement ({no_improve_count}/{plateau_patience})")

                    # Plateau → stop
                    if no_improve_count >= plateau_patience:
                        print(f"\n⏹  Plateau reached — stopping early at step {global_step}.")
                        print(f"   Best val_loss : {best_val_loss:.4f}")
                        print(f"   Best adapter  : {adapter_dir / 'best'}")
                        return

        if global_step >= total_steps:
            break

    print(f"\n✅ Finished tier='{args.tier}'")
    print(f"   Best adapter saved at : {adapter_dir / 'best'}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",        required=True, choices=["high", "mid", "base"],
                        help="Which data tier to train on")
    parser.add_argument("--base_model",  default="SayedShaun/bengali-whisper-medium",
                        help="HuggingFace model ID for base Whisper")
    parser.add_argument("--data_dir",    default="/content/drive/MyDrive/hoichoi/data")
    parser.add_argument("--adapter_dir", default="/content/drive/MyDrive/hoichoi/adapters")
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Per-device batch size (reduce to 4 if OOM on T4)")
    parser.add_argument("--grad_accum",  type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--max_steps",   type=int, default=500,
                        help="Max training steps before forced stop")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--no_drive",    action="store_true")
    args = parser.parse_args()

    install_if_colab()
    if not args.no_drive:
        try_mount_drive()

    train(args)


if __name__ == "__main__":
    main()
