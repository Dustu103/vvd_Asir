"""
src/timed_text.py — Industry Broadcast Timed-Text Formatter & Cue Segmenter
===========================================================================
Problem 2: Automated Bengali Subtitle & Closed-Caption Pipeline with Speaker Diarization
Track: Speech & Language / Media Localisation

Standards Compliance (Netflix / BBC / Hoichoi Timed-Text Specs):
1. Reading Speed (CPS): Strictly <= 17-20 Characters Per Second.
2. Line Length (CPL): Maximum 37-42 characters per line.
3. Cue Height: Maximum 2 lines per subtitle cue.
4. Duration Constraints: Minimum 0.833s (20 frames @ 24fps), Maximum 6.0s-7.0s.
5. Inter-Cue Gap: Minimum 2 frames (~80ms) to prevent perceptual persistence flicker.
6. Shot Boundary Adherence: Subtitles never cross a hard visual cut inappropriately;
   snaps to cut if within 3 frames (~125ms).
7. Format Exporters:
   - WebVTT (.vtt) with voice tags <v SPEAKER> and sound events.
   - SubRip (.srt) for translated subtitles.
"""

import math
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


@dataclass
class TimedCue:
    cue_id: int
    start_sec: float
    end_sec: float
    text: str
    speaker_id: str
    language: str = "bn"          # "bn", "en", "hi"
    is_sound_event: bool = False
    cps: float = 0.0
    cpl_list: List[int] = field(default_factory=list)
    line_count: int = 1
    qc_flags: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.001, self.end_sec - self.start_sec)


class TimedTextFormatter:
    """
    Formats raw transcripts into broadcast-grade subtitle cues adhering to
    strict readability and timed-text industry rules.
    """

    def __init__(
        self,
        max_cps: float = 17.5,
        max_cpl: int = 38,
        max_lines: int = 2,
        min_duration_sec: float = 0.833,
        max_duration_sec: float = 6.5,
        min_gap_sec: float = 0.080,    # 2 frames at 25fps (~80ms)
        snap_shot_threshold_sec: float = 0.125, # 3 frames
    ):
        self.max_cps = max_cps
        self.max_cpl = max_cpl
        self.max_lines = max_lines
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_gap_sec = min_gap_sec
        self.snap_shot_threshold_sec = snap_shot_threshold_sec

    def wrap_lines_linguistically(self, text: str, max_cpl: Optional[int] = None) -> str:
        """
        Splits text into at most 2 balanced lines, preferring natural punctuation
        breaks (।, ?, !, ,, conjunctions) over random word wraps.
        """
        max_cpl = max_cpl or self.max_cpl
        text = unicodedata.normalize("NFC", text.strip())
        if len(text) <= max_cpl:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        # Best split search: balance lines while respecting natural punctuation
        best_split_idx = len(words) // 2
        min_penalty = float("inf")

        for i in range(1, len(words)):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])

            len1, len2 = len(line1), len(line2)
            penalty = abs(len1 - len2)

            # High penalty if exceeding max_cpl
            if len1 > max_cpl:
                penalty += (len1 - max_cpl) * 10
            if len2 > max_cpl:
                penalty += (len2 - max_cpl) * 10

            # Bonus (reduced penalty) for punctuation at end of line 1
            if line1[-1] in ["।", "?", "!", ",", ";", ":", "-"]:
                penalty -= 15

            if penalty < min_penalty:
                min_penalty = penalty
                best_split_idx = i

        line1 = " ".join(words[:best_split_idx])
        line2 = " ".join(words[best_split_idx:])

        return f"{line1}\n{line2}"

    def snap_to_shot_boundaries(
        self,
        start_sec: float,
        end_sec: float,
        shot_cuts: List[float],
    ) -> Tuple[float, float, List[str]]:
        """
        Snaps cue boundaries to nearby visual shot cuts if within threshold,
        and flags if a cue awkwardly straddles an unsnapped shot cut.
        """
        flags = []
        new_start = start_sec
        new_end = end_sec

        for cut in shot_cuts:
            # Snap start to cut
            if abs(new_start - cut) <= self.snap_shot_threshold_sec:
                new_start = cut
            # Snap end to cut (with min gap buffer)
            elif abs(new_end - cut) <= self.snap_shot_threshold_sec:
                new_end = max(new_start + self.min_duration_sec, cut - self.min_gap_sec)
            # Straddle check: cut falls strictly inside the cue
            elif (new_start + 0.25) < cut < (new_end - 0.25):
                flags.append(f"STRADDLES_SHOT_CUT_AT_{cut:.2f}s")

        return round(new_start, 3), round(new_end, 3), flags

    def segment_transcript(
        self,
        raw_items: List[Dict],       # [{"start": float, "end": float, "text": str, "speaker": str, "is_event": bool}]
        shot_cuts: Optional[List[float]] = None,
        language: str = "bn",
    ) -> List[TimedCue]:
        """
        Takes raw timestamped items and formats them into strict broadcast-compliant cues.
        """
        shot_cuts = shot_cuts or []
        cues: List[TimedCue] = []

        for item in raw_items:
            raw_text = unicodedata.normalize("NFC", item.get("text", "").strip())
            if not raw_text:
                continue

            start_t = float(item["start"])
            end_t = float(item["end"])
            speaker = item.get("speaker", "SPEAKER_01")
            is_event = item.get("is_event", False)

            # Ensure minimum duration
            if (end_t - start_t) < self.min_duration_sec:
                end_t = start_t + self.min_duration_sec

            # Wrap into broadcast-friendly lines
            formatted_text = self.wrap_lines_linguistically(raw_text)
            lines = formatted_text.split("\n")
            cpl_list = [len(l) for l in lines]
            line_count = len(lines)

            # Shot cut snapping & straddle audit
            snapped_start, snapped_end, shot_flags = self.snap_to_shot_boundaries(start_t, end_t, shot_cuts)

            # Maintain inter-cue gap with previous cue
            if cues:
                prev_end = cues[-1].end_sec
                if snapped_start < (prev_end + self.min_gap_sec):
                    snapped_start = prev_end + self.min_gap_sec
                    if snapped_end <= snapped_start:
                        snapped_end = snapped_start + self.min_duration_sec

            # Calculate reading speed (CPS)
            char_count = sum(cpl_list)
            dur = max(0.001, snapped_end - snapped_start)
            cps = round(char_count / dur, 1)

            # Rule violation audits
            cue_flags = list(shot_flags)
            if cps > self.max_cps:
                cue_flags.append(f"CPS_EXCEEDED ({cps} > {self.max_cps})")
            if any(l_len > self.max_cpl for l_len in cpl_list):
                cue_flags.append(f"CPL_EXCEEDED (max: {max(cpl_list)} > {self.max_cpl})")
            if line_count > self.max_lines:
                cue_flags.append(f"LINES_EXCEEDED ({line_count} > {self.max_lines})")
            if dur > self.max_duration_sec:
                cue_flags.append(f"DURATION_TOO_LONG ({dur:.1f}s > {self.max_duration_sec}s)")

            cues.append(
                TimedCue(
                    cue_id=len(cues) + 1,
                    start_sec=snapped_start,
                    end_sec=snapped_end,
                    text=formatted_text,
                    speaker_id=speaker,
                    language=language,
                    is_sound_event=is_event,
                    cps=cps,
                    cpl_list=cpl_list,
                    line_count=line_count,
                    qc_flags=cue_flags,
                )
            )

        return cues

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Formats seconds into WebVTT timestamp: HH:MM:SS.mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Formats seconds into SubRip timestamp: HH:MM:SS,mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

    def export_webvtt(self, cues: List[TimedCue], output_path: Union[str, Path]) -> str:
        """
        Exports broadcast-grade WebVTT (.vtt) file with speaker voice tags <v SPEAKER>.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["WEBVTT", "Kind: captions", "Language: bn", ""]

        for cue in cues:
            start_str = self._format_timestamp_vtt(cue.start_sec)
            end_str = self._format_timestamp_vtt(cue.end_sec)

            lines.append(str(cue.cue_id))
            lines.append(f"{start_str} --> {end_str}")

            if cue.is_sound_event:
                # Closed Caption sound event formatting
                lines.append(cue.text)
            else:
                # Speaker voice tag per WebVTT standard
                lines.append(f"<v {cue.speaker_id}>{cue.text}")
            lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)

    def export_srt(self, cues: List[TimedCue], output_path: Union[str, Path]) -> str:
        """
        Exports standard SubRip (.srt) file with speaker labels for translated subtitles.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for cue in cues:
            start_str = self._format_timestamp_srt(cue.start_sec)
            end_str = self._format_timestamp_srt(cue.end_sec)

            lines.append(str(cue.cue_id))
            lines.append(f"{start_str} --> {end_str}")

            if cue.is_sound_event:
                lines.append(cue.text)
            else:
                # SubRip speaker label prefix (industry standard)
                lines.append(f"[{cue.speaker_id}] {cue.text}")
            lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)
