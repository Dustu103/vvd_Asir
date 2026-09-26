"""
problem_2_caption_diarization/src/diarization.py
=================================================
Persistent Speaker Diarization & Cast List Vocative Matcher

Real model used:
  Speaker Embeddings : speechbrain/spkrec-ecapa-voxceleb
                       (256-dim ECAPA-TDNN, no token needed, ~100MB)
  Clustering         : Agglomerative hierarchical with cosine distance
  Fallback           : Custom 64-dim spectral fingerprint (pure numpy)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeakerTurn:
    turn_id: int
    start_sec: float
    end_sec: float
    speaker_id: str              # e.g. "SPEAKER_01" or resolved "FELUDA"
    confidence: float = 0.90
    cluster_label: int = 0
    resolved_name: Optional[str] = None


@dataclass
class CastMember:
    name: str
    aliases: List[str]
    gender: Optional[str] = None
    role: Optional[str] = None


# ---------------------------------------------------------------------------
# SpeechBrain ECAPA-TDNN speaker embedding extractor
# ---------------------------------------------------------------------------
class EcapaTDNNEmbedder:
    """
    Wraps speechbrain/spkrec-ecapa-voxceleb for 256-dim speaker embeddings.
    No HuggingFace token required — weights are public.
    Automatic fallback to spectral fingerprint if speechbrain not installed.
    """

    MODEL_HUB = "speechbrain/spkrec-ecapa-voxceleb"
    EMBED_DIM  = 192    # ECAPA-TDNN output dimension

    def __init__(self, device: str = "cpu"):
        self._model  = None
        self._device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from speechbrain.pretrained import EncoderClassifier
            print(f"[Diarize] Loading SpeechBrain ECAPA-TDNN from {self.MODEL_HUB}...")
            self._model = EncoderClassifier.from_hparams(
                source=self.MODEL_HUB,
                run_opts={"device": self._device},
                savedir=f"/tmp/speechbrain_ecapa",
            )
            print("[Diarize] ECAPA-TDNN ready.")
        except Exception as ex:
            print(f"[Diarize] SpeechBrain unavailable ({ex}). Using spectral fingerprint fallback.")

    def embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Returns L2-normalised speaker embedding vector.
        Shape: (EMBED_DIM,)
        """
        self._load()

        if self._model is not None and len(audio) >= int(sr * 0.5):
            try:
                import torch
                # SpeechBrain expects (batch, samples) float32
                wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    emb = self._model.encode_batch(wav)          # (1, 1, EMBED_DIM)
                vec = emb.squeeze().cpu().numpy()
                norm = np.linalg.norm(vec) + 1e-12
                return (vec / norm).astype(np.float32)
            except Exception as ex:
                print(f"  [Diarize] ECAPA inference error ({ex}). Falling back.")

        return self._spectral_fallback(audio, sr)

    @staticmethod
    def _spectral_fallback(audio: np.ndarray, sr: int, dim: int = 64) -> np.ndarray:
        """64-dim multi-band spectral fingerprint when ECAPA unavailable."""
        if len(audio) < int(sr * 0.15):
            return np.zeros(dim, dtype=np.float32)

        windowed = audio * np.hanning(len(audio))
        fft      = np.abs(np.fft.rfft(windowed)) + 1e-12
        n_bins   = len(fft)
        n_bands  = 32
        band_sz  = max(1, n_bins // n_bands)

        mel = np.array([
            np.log(np.mean(fft[b * band_sz: min(n_bins, (b+1) * band_sz)]) + 1e-6)
            for b in range(n_bands)
        ], dtype=np.float32)

        sub_windows = np.array_split(audio, 8)
        sub_rms = np.array([np.sqrt(np.mean(w**2) + 1e-12) for w in sub_windows], dtype=np.float32)
        sub_zcr = np.array([
            float(np.mean(np.abs(np.diff(np.sign(w)))) / 2) for w in sub_windows
        ], dtype=np.float32)

        vec = np.concatenate([mel, sub_rms, sub_zcr, [mel.mean(), mel.std()],
                               mel[:14]])[:dim].astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------------
def _agglomerative_cluster(
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
) -> List[int]:
    """
    Bottom-up agglomerative clustering with cosine distance.
    Returns cluster label per embedding.
    """
    n = len(embeddings)
    clusters  = list(range(n))
    centroids = [embeddings[i].copy() for i in range(n)]

    while True:
        unique = list(set(clusters))
        if len(unique) <= 1:
            break

        best_dist, best_pair = float("inf"), None
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                c1, c2 = unique[i], unique[j]
                sim  = np.dot(centroids[c1], centroids[c2])
                dist = 1.0 - float(sim)
                if dist < best_dist:
                    best_dist, best_pair = dist, (c1, c2)

        if best_pair is None or best_dist >= distance_threshold:
            break

        tgt, src = best_pair
        for idx in range(n):
            if clusters[idx] == src:
                clusters[idx] = tgt
        members = [embeddings[k] for k in range(n) if clusters[k] == tgt]
        centroids[tgt] = np.mean(members, axis=0)
        norm = np.linalg.norm(centroids[tgt]) + 1e-12
        centroids[tgt] /= norm

    return clusters


# ---------------------------------------------------------------------------
# Main diarization engine
# ---------------------------------------------------------------------------
class SpeakerDiarizationEngine:
    """
    ECAPA-TDNN speaker embedding + agglomerative clustering diarizer.
    Falls back to spectral fingerprints if SpeechBrain unavailable.
    """

    # Hoichoi show cast registry (vocative resolution)
    HOICHOI_CAST_REGISTRY: Dict[str, List[CastMember]] = {
        "feluda": [
            CastMember("FELUDA",   ["ফেলুদা", "প্রদোষ", "মিত্র", "feluda", "pradosh"]),
            CastMember("TOPSHE",   ["তোপসে", "তপেশ", "topshe", "tapesh"]),
            CastMember("JATAYU",   ["জটায়ু", "লালমোহন", "jatayu", "lalmohan"]),
            CastMember("MAGANLAL", ["মগনলাল", "শেঠ", "maganlal"]),
        ],
        "mohanagar": [
            CastMember("OC_HARUN", ["হারুন", "ওসি", "স্যার", "harun", "oc"]),
            CastMember("AFNAN",    ["আফনান", "চৌধুরী", "afnan"]),
            CastMember("MOLOY",    ["মলয়", "moloy"]),
        ],
        "mandaar": [
            CastMember("MANDAAR",  ["মান্দার", "mandaar"]),
            CastMember("LAILI",    ["লাইলি", "laili"]),
            CastMember("BONKA",    ["বোংকা", "bonka"]),
        ],
        "bhojon_bilashi": [
            CastMember("HOST",  ["শেফ", "দাদা", "host", "chef"]),
            CastMember("GUEST", ["অতিথি", "guest"]),
        ],
        "money_honey": [
            CastMember("SHEKHOR", ["শেখর", "shekhor"]),
            CastMember("SANJU",   ["সঞ্জু", "sanju"]),
        ],
    }

    def __init__(
        self,
        sample_rate: int = 16000,
        distance_threshold: float = 0.35,
        min_turn_duration_sec: float = 0.80,
        device: str = "cpu",
    ):
        self.sample_rate          = sample_rate
        self.distance_threshold   = distance_threshold
        self.min_turn_duration_sec = min_turn_duration_sec
        self._embedder = EcapaTDNNEmbedder(device=device)

    def diarize_segments(
        self,
        full_audio: np.ndarray,
        speech_intervals: List[Tuple[float, float]],
    ) -> List[SpeakerTurn]:
        """
        Extracts ECAPA-TDNN embeddings for each speech segment,
        clusters them into persistent speaker IDs, and applies
        temporal smoothing to suppress short flickering turns.
        """
        if not speech_intervals:
            return []

        valid_intervals, embeddings = [], []

        for start_sec, end_sec in speech_intervals:
            s = int(start_sec * self.sample_rate)
            e = int(end_sec   * self.sample_rate)
            clip = full_audio[s:e]
            if len(clip) >= int(self.sample_rate * 0.2):
                emb = self._embedder.embed(clip, self.sample_rate)
                embeddings.append(emb)
                valid_intervals.append((start_sec, end_sec))

        if not valid_intervals:
            return []

        emb_arr  = np.array(embeddings)
        clusters = _agglomerative_cluster(emb_arr, self.distance_threshold)

        # Map cluster IDs to sequential SPEAKER_XX labels (most frequent first)
        freq_order = sorted(set(clusters), key=lambda c: clusters.count(c), reverse=True)
        spk_map    = {c: f"SPEAKER_{i+1:02d}" for i, c in enumerate(freq_order)}

        raw_turns: List[SpeakerTurn] = [
            SpeakerTurn(
                turn_id=idx + 1,
                start_sec=round(st, 3),
                end_sec=round(en, 3),
                speaker_id=spk_map[clusters[idx]],
                cluster_label=clusters[idx],
                confidence=0.92,
            )
            for idx, (st, en) in enumerate(valid_intervals)
        ]

        return self._smooth_turns(raw_turns)

    def _smooth_turns(self, turns: List[SpeakerTurn]) -> List[SpeakerTurn]:
        """Bridge same-speaker gaps < 400ms; merge micro-turns < 800ms into predecessor."""
        out: List[SpeakerTurn] = []
        for turn in turns:
            if not out:
                out.append(turn)
                continue

            last = out[-1]
            dur  = turn.end_sec - turn.start_sec

            if turn.speaker_id == last.speaker_id and (turn.start_sec - last.end_sec) < 0.40:
                last.end_sec = turn.end_sec          # bridge same-speaker gap
            elif dur < self.min_turn_duration_sec and (turn.start_sec - last.end_sec) < 0.25:
                last.end_sec = turn.end_sec          # absorb micro-flicker
            else:
                turn.turn_id = len(out) + 1
                out.append(turn)
        return out

    def resolve_cast_vocatives(
        self,
        turns: List[SpeakerTurn],
        transcripts: List[str],
        show_name_hint: Optional[str] = None,
    ) -> List[SpeakerTurn]:
        """
        Scans transcripts for vocative cues (e.g. 'তোপসে, এদিকে এসো') to
        upgrade SPEAKER_XX labels to real character names.
        """
        if not turns or not transcripts:
            return turns

        # Select cast list
        cast_list: List[CastMember] = []
        if show_name_hint:
            for key, members in self.HOICHOI_CAST_REGISTRY.items():
                if key in show_name_hint.lower():
                    cast_list = members
                    break
        if not cast_list:
            cast_list = self.HOICHOI_CAST_REGISTRY["feluda"]

        # Evidence accumulator: speaker_id → {character_name: vote_count}
        evidence: Dict[str, Dict[str, int]] = {}

        for i, turn in enumerate(turns):
            if i >= len(transcripts):
                break
            text = transcripts[i].lower()
            spk  = turn.speaker_id
            evidence.setdefault(spk, {})

            for cast in cast_list:
                for alias in cast.aliases:
                    if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text):
                        # This speaker is addressing `cast.name`
                        others = [t.speaker_id for t in turns if t.speaker_id != spk]
                        if others:
                            addressee = max(set(others), key=others.count)
                            evidence.setdefault(addressee, {})
                            evidence[addressee][cast.name] = (
                                evidence[addressee].get(cast.name, 0) + 1
                            )

        for spk, votes in evidence.items():
            if votes:
                winner = max(votes, key=votes.get)
                for t in turns:
                    if t.speaker_id == spk:
                        t.resolved_name = winner
                        t.speaker_id    = winner

        return turns
