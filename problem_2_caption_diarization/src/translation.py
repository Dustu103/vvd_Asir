"""
problem_2_caption_diarization/src/translation.py
=================================================
Multilingual Subtitle Translator  (Bengali -> English & Hindi)

Engine priority per language pair:
  BN -> EN : Helsinki-NLP/opus-mt-bn-en  (neural, ~300 MB)
  BN -> HI : facebook/nllb-200-distilled-600M  (neural, Indic pair, ~1.2 GB)
  Fallback  : Embedded lexical engine — always runs offline, zero external deps
"""

import re
import unicodedata
from typing import List, Dict, Optional
from .timed_text import TimedCue


# ---------------------------------------------------------------------------
# Bilingual sound event tags (Bengali CC -> EN / HI)
# ---------------------------------------------------------------------------
SOUND_EVENT_MAP: Dict[str, tuple] = {
    "[মিউজিক]":        ("[MUSIC PLAYING]",  "[संगीत]"),
    "[মিউজিক বাজছে]":  ("[MUSIC PLAYING]",  "[संगीत बज रहा है]"),
    "[হাসি]":           ("[LAUGHTER]",       "[हंसी]"),
    "[হাততালি]":        ("[APPLAUSE]",       "[तालियां]"),
    "[গুলির শব্দ]":     ("[GUNSHOT]",        "[गोली की आवाज]"),
    "[ফোনের রিং]":      ("[PHONE RINGING]",  "[फोन की घंटी]"),
    "[পায়ের আওয়াজ]":  ("[FOOTSTEPS]",      "[कदमों की आहट]"),
    "[নীরবতা]":         ("[SILENCE]",        "[सन्नाटा]"),
}

# ---------------------------------------------------------------------------
# Fallback lexical engines (phrases first, then words)
# ---------------------------------------------------------------------------
BN_EN_PHRASES = {
    "ঠিক আছে": "All right",
    "কেমন আছো": "How are you",
    "কি খবর": "What's the news",
    "এদিকে এসো": "Come here",
    "আমি জানি না": "I don't know",
    "আমি জানি": "I know",
    "কী হয়েছে": "What happened",
}
BN_EN_WORDS = {
    "নমস্কার": "Hello", "হ্যালো": "Hello", "হ্যাঁ": "Yes", "না": "No",
    "অফিস": "office", "মিটিং": "meeting", "কোথায়": "where",
    "ধন্যবাদ": "Thank you", "জলদি": "quickly", "চলো": "Let's go",
    "পুলিশ": "police", "থানা": "police station", "স্যার": "sir",
    "বাবু": "sir", "দেখো": "Look", "শোনো": "Listen",
    "সাবধান": "Be careful", "খাবার": "food", "টাকা": "money",
    "গাড়ি": "car", "চাবি": "key", "দরজা": "door",
    "বন্ধু": "friend", "কে": "who", "ওখানে": "there",
    "এখানে": "here", "কেন": "why", "কী": "what",
    "কখন": "when", "আসছি": "coming",
}

BN_HI_PHRASES = {
    "ঠিক আছে": "ठीक है",
    "কেমন আছো": "आप कैसे हैं",
    "কি খবর": "क्या खबर है",
    "এদিকে এসো": "इधर आओ",
    "আমি জানি না": "मुझे नहीं पता",
    "আমি জানি": "मुझे पता है",
    "কী হয়েছে": "क्या हुआ",
}
BN_HI_WORDS = {
    "নমস্কার": "नमस्ते", "হ্যালো": "हैलो", "হ্যাঁ": "हाँ", "না": "नहीं",
    "অফিস": "ऑफिस", "মিটিং": "मीटिंग", "কোথায়": "कहाँ",
    "ধন্যবাদ": "धन्यवाद", "জলদি": "जल्दी", "চলো": "चलो",
    "পুলিশ": "पुलिस", "থানা": "थाना", "স্যার": "सर",
    "বাবু": "बाबू", "দেখো": "देखो", "শোনো": "सुनो",
    "সাবধান": "सावधान", "খাবার": "खाना", "টাকা": "पैसे",
    "গাড়ি": "गाड़ी", "চাবি": "चाबी", "দরজা": "दरवाजा",
    "বন্ধু": "दोस्त", "কে": "कौन", "ওখানে": "वहाँ",
    "এখানে": "यहाँ", "কেন": "क्यों", "কী": "क्या",
    "কখন": "कब", "আসছি": "आ रहा हूँ",
}


def _lexical_translate(text: str, target_lang: str) -> str:
    """Pure lexical fallback — runs offline with zero model loading."""
    phrases = BN_EN_PHRASES if target_lang == "en" else BN_HI_PHRASES
    words   = BN_EN_WORDS   if target_lang == "en" else BN_HI_WORDS
    out = text
    for phrase, trans in sorted(phrases.items(), key=lambda x: len(x[0]), reverse=True):
        out = out.replace(phrase, trans)
    for word, trans in sorted(words.items(), key=lambda x: len(x[0]), reverse=True):
        out = re.sub(r"(?<!\w)" + re.escape(word) + r"(?!\w)", trans, out)
    return out.strip()


class SubtitleTranslator:
    """
    Translates Bengali subtitle cues into English and Hindi.

    Neural engine priority:
      EN : Helsinki-NLP/opus-mt-bn-en
      HI : facebook/nllb-200-distilled-600M  (src=ben_Beng, tgt=hin_Deva)
    Falls back to embedded lexical engine if models are unavailable.
    """

    _NLLB_BN_TAG  = "ben_Beng"
    _NLLB_HI_TAG  = "hin_Deva"
    _NLLB_EN_TAG  = "eng_Latn"

    def __init__(self, use_neural: bool = True):
        self.use_neural = use_neural
        self._pipe_en   = None   # opus-mt-bn-en pipeline
        self._pipe_nllb = None   # nllb-200 pipeline (for HI, and EN fallback)
        self._loaded    = False

        if self.use_neural:
            self._load_models()

    def _load_models(self):
        if self._loaded:
            return
        self._loaded = True

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("[Trans] Loading Helsinki-NLP/opus-mt-bn-en ...")
            self._tok_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
            self._model_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-bn-en").eval()
            print("[Trans] opus-mt-bn-en ready.")
        except Exception as e:
            print(f"[Trans] opus-mt-bn-en unavailable ({e}). Using lexical EN fallback.")

        try:
            from transformers import pipeline as hf_pipeline
            # BN -> HI : nllb-200 (covers all Indic pairs)
            print("[Trans] Loading facebook/nllb-200-distilled-600M ...")
            self._pipe_nllb = hf_pipeline(
                "text2text-generation",
                model="facebook/nllb-200-distilled-600M",
            )
            print("[Trans] nllb-200 ready.")
        except Exception as e:
            print(f"[Trans] nllb-200 unavailable ({e}). Using lexical HI fallback.")

    def translate_text(self, text: str, target_lang: str = "en") -> str:
        """Translates a single line of Bengali text."""
        import torch
        text = unicodedata.normalize("NFC", text.strip())
        if not text:
            return ""

        # Pass-through sound event tags
        for bn_tag, (en_tag, hi_tag) in SOUND_EVENT_MAP.items():
            if bn_tag in text:
                return en_tag if target_lang == "en" else hi_tag

        # Neural engines
        if self.use_neural:
            if target_lang == "en" and getattr(self, "_model_en", None):
                try:
                    inputs = self._tok_en([text], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        out = self._model_en.generate(**inputs, max_length=256)
                    decoded = self._tok_en.batch_decode(out, skip_special_tokens=True)[0].strip()
                    if decoded:
                        return decoded
                except Exception:
                    pass

            if target_lang == "hi" and self._pipe_nllb:
                try:
                    # nllb needs explicit tgt_lang at inference time
                    res = self._pipe_nllb(
                        text,
                        forced_bos_token_id=self._pipe_nllb.tokenizer.convert_tokens_to_ids(
                            self._NLLB_HI_TAG
                        ),
                        max_length=256,
                    )
                    out = res[0].get("translation_text", "").strip()
                    if out:
                        return out
                except Exception:
                    pass

        # Lexical fallback
        return _lexical_translate(text, target_lang)

    def translate_cues(
        self,
        bengali_cues: List[TimedCue],
        target_lang: str = "en",
    ) -> List[TimedCue]:
        """
        Translates a list of Bengali TimedCues, preserving timestamps,
        speaker attribution, and sound event tags intact.
        """
        out_cues: List[TimedCue] = []

        for cue in bengali_cues:
            lines      = cue.text.split("\n")
            trans_lines = [self.translate_text(line, target_lang=target_lang) for line in lines]
            trans_text  = "\n".join(trans_lines)

            dur = max(0.001, cue.end_sec - cue.start_sec)
            cps = round(sum(len(l) for l in trans_lines) / dur, 1)

            out_cues.append(
                TimedCue(
                    cue_id=cue.cue_id,
                    start_sec=cue.start_sec,
                    end_sec=cue.end_sec,
                    text=trans_text,
                    speaker_id=cue.speaker_id,
                    language=target_lang,
                    is_sound_event=cue.is_sound_event,
                    cps=cps,
                    cpl_list=[len(l) for l in trans_lines],
                    line_count=len(trans_lines),
                )
            )

        return out_cues
