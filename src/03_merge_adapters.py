"""
03_merge_adapters.py  —  TIES-merge the 3 tier adapters into one final adapter
===============================================================================
Run after all 3 adapters are trained:

    !python src/03_merge_adapters.py

Input  : /content/drive/MyDrive/hoichoi/adapters/{high,mid,base}/best/
Output : /content/drive/MyDrive/hoichoi/adapters/merged/

Strategy: TIES merging (Trim, Elect, Sum) with density=0.7
  - Better than simple weighted average for adapters trained on different distributions
  - Prunes conflicting parameter directions before combining
  - Bias weights toward 'high' tier (most switch-dense)
"""

import os, sys, json, argparse
from pathlib import Path

def try_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted")
    except Exception:
        print("ℹ️  Not in Colab — skipping Drive mount.")

def install_if_colab():
    try:
        import google.colab  # noqa
        os.system("pip install -q transformers>=4.40.0 peft>=0.10.0")
    except ImportError:
        pass


def check_adapter_exists(path: Path, tier: str) -> bool:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    # safetensors or bin
    has_weights = (path / "adapter_model.safetensors").exists() or \
                  (path / "adapter_model.bin").exists()
    has_config  = (path / "adapter_config.json").exists()
    if not (has_weights and has_config):
        print(f"  ⚠️  Adapter '{tier}' not found or incomplete at {path}")
        return False
    print(f"  ✅ Adapter '{tier}' found at {path}")
    return True


def merge(args):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel, LoraConfig

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    adapter_root = Path(args.adapter_dir)
    merged_dir   = adapter_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # ── Verify all adapters are ready ────────────────────────────────────────
    tier_paths = {
        "high": adapter_root / "high" / "best",
        "mid":  adapter_root / "mid"  / "best",
        "base": adapter_root / "base" / "best",
    }

    print("\n[1/4] Checking adapter availability …")
    available = {}
    for tier, path in tier_paths.items():
        if check_adapter_exists(path, tier):
            available[tier] = path

    if len(available) < 2:
        print(
            "\n❌ Need at least 2 trained adapters to merge.\n"
            "   Run 02_train_adapter.py --tier high (and mid/base) first."
        )
        sys.exit(1)

    if len(available) < 3:
        missing = [t for t in ["high", "mid", "base"] if t not in available]
        print(f"\n⚠️  Merging with only {len(available)} adapters (missing: {missing})")
        print("   Proceeding — you can re-merge after the missing tier finishes.\n")

    # ── Weights: bias toward high-switch adapter ──────────────────────────────
    # Default: high=0.5, mid=0.3, base=0.2
    # Normalized to sum to 1.0 across available tiers
    raw_weights = {"high": 0.5, "mid": 0.3, "base": 0.2}
    filtered    = {t: raw_weights[t] for t in available}
    total       = sum(filtered.values())
    weights     = {t: w / total for t, w in filtered.items()}
    print(f"  Merge weights (normalized): {weights}")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading base model: {args.base_model} …")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.config.use_cache = False

    # ── Load first adapter as base PEFT model ─────────────────────────────────
    print("\n[3/4] Loading adapters into PEFT model …")
    tiers_ordered = [t for t in ["high", "mid", "base"] if t in available]
    first_tier    = tiers_ordered[0]

    peft_model = PeftModel.from_pretrained(
        model,
        str(available[first_tier]),
        adapter_name=first_tier,
    )

    # Load remaining adapters
    for tier in tiers_ordered[1:]:
        peft_model.load_adapter(str(available[tier]), adapter_name=tier)
        print(f"  Loaded adapter: {tier}")

    # ── TIES merge ────────────────────────────────────────────────────────────
    print(f"\n[4/4] Running TIES merge (density={args.density}) …")
    peft_model.add_weighted_adapter(
        adapters         = tiers_ordered,
        weights          = [weights[t] for t in tiers_ordered],
        adapter_name     = "merged",
        combination_type = "ties",   # TIES: trim conflicting, elect direction, sum
        density          = args.density,
    )
    peft_model.set_adapter("merged")

    # ── Save merged adapter ───────────────────────────────────────────────────
    peft_model.save_pretrained(str(merged_dir))

    # Also save the processor from the first available adapter
    try:
        processor = WhisperProcessor.from_pretrained(str(available[first_tier]))
        processor.save_pretrained(str(merged_dir))
    except Exception as e:
        print(f"  ⚠️  Could not save processor: {e}")

    # Save merge metadata
    meta = {
        "base_model":    args.base_model,
        "adapters_used": tiers_ordered,
        "weights":       weights,
        "density":       args.density,
        "combination":   "ties",
    }
    (merged_dir / "merge_config.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅ Merged adapter saved → {merged_dir}")
    print(f"   Next: run 04_evaluate.py to score this against your baseline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default="SayedShaun/bengali-whisper-medium")
    parser.add_argument("--adapter_dir", default="/content/drive/MyDrive/hoichoi/adapters")
    parser.add_argument("--density",     type=float, default=0.7,
                        help="TIES density: fraction of params to keep (0.7 = trim bottom 30%%)")
    parser.add_argument("--no_drive",    action="store_true")
    args = parser.parse_args()

    install_if_colab()
    if not args.no_drive:
        try_mount_drive()

    merge(args)


if __name__ == "__main__":
    main()
