"""
ApexStudent Tokenizer: Custom-trained Byte-Pair Encoding (BPE) subword tokenizer.
Trained 100% from scratch on our task domain texts without pretrained tokenizer weights.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class ApexTokenizer:
    """
    Production-grade fast subword tokenizer for ApexStudent.
    Wraps HuggingFace tokenizers library or provides high-speed native fallback.
    """

    SPECIAL_TOKENS = [
        "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]",
        "[NOUL]", "[CHOICE]", "[SCORE]", "[VERIFY]", "[EVIDENCE]"
    ]

    def __init__(self, tokenizer_obj=None, vocab_file: Optional[str] = None):
        self._tok = tokenizer_obj
        if tokenizer_obj is None and vocab_file is None:
            candidates = [str(Path(__file__).resolve().parent / "tokenizer_weights" / "tokenizer.json")]
            for c in candidates:
                if os.path.exists(c):
                    vocab_file = c
                    break

        if self._tok is None and vocab_file is not None and os.path.exists(vocab_file):
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(vocab_file)

        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"

        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.unk_token_id = 4

        if self._tok is not None:
            self._update_token_ids()

        try:
            from .multilingual_engine import MultilingualEngine
        except (ImportError, ValueError):
            from apex_decision.models.apex_student.multilingual_engine import MultilingualEngine
        self.multilingual = MultilingualEngine(vocab_size=self.vocab_size)

    def _update_token_ids(self):
        try:
            self.pad_token_id = self._tok.token_to_id(self.pad_token) or 0
            self.cls_token_id = self._tok.token_to_id(self.cls_token) or 1
            self.sep_token_id = self._tok.token_to_id(self.sep_token) or 2
            self.mask_token_id = self._tok.token_to_id(self.mask_token) or 3
            self.unk_token_id = self._tok.token_to_id(self.unk_token) or 4
        except Exception:
            pass

    @property
    def vocab_size(self) -> int:
        if self._tok is not None:
            return self._tok.get_vocab_size()
        return 4096

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        norm_text = self.multilingual.normalize_text(text)
        script = self.multilingual.detect_script(norm_text)
        
        if script == "latin" and self._tok is not None:
            enc = self._tok.encode(norm_text)
            ids = [min(max(0, i), self.vocab_size - 1) for i in enc.ids]
        else:
            base_enc = (lambda t: self._tok.encode(t).ids) if self._tok is not None else None
            ids = self.multilingual.encode_multilingual(norm_text, base_tokenizer_encode=base_enc)

        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self._tok is None:
            return "".join([chr(i % 128) for i in ids if i >= 32])
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            att = [1] * len(ids)
            if padding and max_length and len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                att = att + [0] * pad_len
            res = {"input_ids": ids, "attention_mask": att}
        else:
            batch_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
            if truncation and max_length:
                batch_ids = [b[:max_length] for b in batch_ids]
            if padding:
                actual_max = max_length if max_length else max(len(b) for b in batch_ids)
                padded_ids = []
                attention_masks = []
                for b in batch_ids:
                    pad_len = max(0, actual_max - len(b))
                    padded_ids.append(b + [self.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(b) + [0] * pad_len)
                batch_ids = padded_ids
                att = attention_masks
            else:
                att = [[1] * len(b) for b in batch_ids]
            res = {"input_ids": batch_ids, "attention_mask": att}

        if return_tensors == "pt":
            import torch
            return {k: torch.tensor(v, dtype=torch.long) for k, v in res.items()}
        return res

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self._tok is not None:
            self._tok.save(os.path.join(save_dir, "tokenizer.json"))
        cfg = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "ApexTokenizer":
        tokenizer_json = os.path.join(save_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json):
            from tokenizers import Tokenizer
            tok_obj = Tokenizer.from_file(tokenizer_json)
            return cls(tok_obj)
        return cls()


def train_apex_tokenizer(
    corpus_texts: List[str],
    vocab_size: int = 16384,
    save_dir: Optional[str] = None
) -> ApexTokenizer:
    """
    Train a fresh BPE tokenizer completely from scratch on the provided corpus.
    Uses ByteLevel pre-tokenization with special tokens.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=ApexTokenizer.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False
    )

    tokenizer.train_from_iterator(corpus_texts, trainer=trainer)
    tok_wrapper = ApexTokenizer(tokenizer)

    if save_dir:
        tok_wrapper.save_pretrained(save_dir)

    return tok_wrapper
