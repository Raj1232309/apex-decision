"""
Enhanced Evidence Retrieval Module (v2) for ApexDecision.

Features:
1. Sentence Segmentation preserving technical, medical, scientific, and legal nomenclature.
2. Multi-sentence candidate formation (individual sentences + contiguous 2-sentence windows).
3. Hybrid BM25-style lexical scoring + Entity/Numeric/Date salience boost.
4. Multi-sentence reasoning support: handles claims requiring cross-sentence fact synthesis.
5. Guarantees true evidence selection for long context sources (no short-source pass-through bypass).
"""

import re
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter


class SentenceSegmenterV2:
    """
    Robust sentence segmenter for multi-sentence and paragraph-length technical sources.
    """
    ABBREVIATIONS = {
        "e.g.", "i.e.", "dr.", "prof.", "vs.", "mr.", "mrs.", "ms.", "art.", "sec.", "fig.", "no.", "al.",
        "inc.", "ltd.", "corp.", "co.", "approx.", "u.s.", "u.k.", "e.u.", "vol.", "dept.", "est."
    }

    @classmethod
    def segment(cls, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        tokens = text.split(" ")
        sentences = []
        current = []

        for token in tokens:
            current.append(token)
            if any(token.endswith(p) for p in [".", "!", "?"]):
                clean = token.lower().strip("()\"'")
                is_abbrev = clean in cls.ABBREVIATIONS or re.match(r'^\d+\.$', clean) or (len(clean) == 2 and clean.endswith("."))
                if not is_abbrev:
                    sentences.append(" ".join(current))
                    current = []

        if current:
            sentences.append(" ".join(current))

        cleaned = [s.strip() for s in sentences if len(s.strip()) > 3]
        return cleaned if cleaned else [text]


class HybridEvidenceRetrieverV2:
    """
    Retrieves the most salient evidence spans for a given claim.
    Ranks individual sentences, 2-sentence contiguous pairs, and combines them
    to provide the NLI cross-encoder with targeted, high-precision context.
    """
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "is", "was", "were", "are", "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "that", "this", "these", "those", "it", "its", "as"
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 1 and t not in cls.STOPWORDS]

    @classmethod
    def extract_salient_features(cls, text: str) -> Dict[str, Set[str]]:
        lower = text.lower()
        numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?%?\b', text))
        dates = set(re.findall(r'\b(?:19|20)\d{2}\b|\bq[1-4]\b', lower))
        entities = set(re.findall(r'\b[A-Z][a-zA-Z0-9_\-]*(?:\s+[A-Z][a-zA-Z0-9_\-]*)*\b', text))
        common_words = {"The", "A", "An", "In", "On", "At", "By", "For", "With", "After", "Before", "During"}
        clean_entities = {e.lower() for e in entities if e not in common_words and len(e) > 1}
        return {
            "numbers": numbers,
            "dates": dates,
            "entities": clean_entities
        }

    @classmethod
    def score_candidate(cls, claim_tokens: List[str], claim_feats: Dict[str, Set[str]], cand_text: str) -> float:
        cand_tokens = cls.tokenize(cand_text)
        if not cand_tokens:
            return 0.0

        # Term frequency overlap (BM25 term match intuition)
        cand_counts = Counter(cand_tokens)
        token_score = 0.0
        for t in claim_tokens:
            if t in cand_counts:
                tf = cand_counts[t]
                token_score += (tf / (tf + 1.2))

        # Salience feature boosts
        cand_feats = cls.extract_salient_features(cand_text)

        num_overlap = len(claim_feats["numbers"].intersection(cand_feats["numbers"]))
        date_overlap = len(claim_feats["dates"].intersection(cand_feats["dates"]))
        ent_overlap = len(claim_feats["entities"].intersection(cand_feats["entities"]))

        total_score = token_score + (1.5 * num_overlap) + (1.0 * date_overlap) + (1.2 * ent_overlap)
        return total_score

    @classmethod
    def retrieve_evidence(
        cls,
        claim: str,
        source: str,
        top_k: int = 2,
        include_contiguous_pairs: bool = True
    ) -> Tuple[List[str], str]:
        """
        Returns:
            - List of selected evidence sentences
            - Joined context string ready for cross-encoder encoding
        """
        sentences = SentenceSegmenterV2.segment(source)
        if not sentences:
            return [], source
        if len(sentences) == 1:
            return sentences, sentences[0]

        claim_tokens = cls.tokenize(claim)
        claim_feats = cls.extract_salient_features(claim)

        # Generate candidates: individual sentences + contiguous 2-sentence windows
        candidates = []
        for idx, s in enumerate(sentences):
            candidates.append({"type": "single", "indices": [idx], "text": s})

        if include_contiguous_pairs and len(sentences) > 2:
            for idx in range(len(sentences) - 1):
                pair_text = f"{sentences[idx]} {sentences[idx+1]}"
                candidates.append({"type": "pair", "indices": [idx, idx+1], "text": pair_text})

        # Score candidates
        scored_candidates = []
        for c in candidates:
            score = cls.score_candidate(claim_tokens, claim_feats, c["text"])
            scored_candidates.append((score, c))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top non-overlapping or highest-coverage sentences
        selected_sentence_indices = set()
        selected_spans = []

        for score, cand in scored_candidates:
            if len(selected_spans) >= top_k:
                break
            # Add indices
            new_indices = [idx for idx in cand["indices"] if idx not in selected_sentence_indices]
            if new_indices:
                for idx in new_indices:
                    selected_sentence_indices.add(idx)
                    selected_spans.append(sentences[idx])

        # Preserve original document ordering of selected sentences
        ordered_spans = [s for idx, s in enumerate(sentences) if idx in selected_sentence_indices]
        if not ordered_spans:
            ordered_spans = sentences[:top_k]

        context_text = " ".join(ordered_spans)
        return ordered_spans, context_text
