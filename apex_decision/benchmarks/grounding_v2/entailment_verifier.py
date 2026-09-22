"""
Unified High-Precision Entailment Verifier (v2) for ApexDecision.

Integrates:
1. Claim Decomposition (atomic proposition splitting).
2. Hybrid Evidence Retrieval (BM25 + dense token/entity/numeric overlap).
3. Structured Semantic Attribute Checks (numbers, entities, dates, causality, quantifiers).
4. Cross-Encoder Entailment Scoring (3-way internal: Entailment, Contradiction, Insufficient Evidence).
5. Conservative Decision Aggregator (all propositions must hold for overall claim support).
6. Microsecond-level Latency Measurement.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .claim_decomposer import ClaimDecomposer, AtomicProposition
from .evidence_retriever_v2 import HybridEvidenceRetrieverV2
from .structured_checks import StructuredClaimChecker, StructuredAlignmentResult


class ApexEntailmentVerifierV2:
    """
    High-Precision Verification Architecture for Grounding and Claim Verification.
    """
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        base_model_name: Optional[str] = None,
        lora_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_structured_checks: bool = True,
        use_decomposition: bool = True,
        temperature: float = 1.0
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_structured_checks = use_structured_checks
        self.use_decomposition = use_decomposition
        self.temperature = max(1e-3, temperature)

        if model is not None and tokenizer is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        elif lora_dir is not None:
            from peft import PeftModel
            base = base_model_name or "answerdotai/ModernBERT-base"
            self.tokenizer = AutoTokenizer.from_pretrained(base)
            base_m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
            self.model = PeftModel.from_pretrained(base_m, lora_dir).to(self.device)
        else:
            path = model_name_or_path or "cross-encoder/nli-deberta-v3-base"
            snapshot_dir = os.getenv("APEX_BENCHMARK_MODEL_PATH", "")
            if path == "cross-encoder/nli-deberta-v3-base" and os.path.exists(snapshot_dir):
                path = snapshot_dir
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=os.path.exists(path))
            self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=os.path.exists(path)).to(self.device)
        self.model.eval()

        # Check model output labels
        cfg = self.model.config
        self.num_labels = getattr(cfg, "num_labels", 3)
        self.id2label = getattr(cfg, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        
        # Determine label indices
        self.entail_idx = None
        self.contra_idx = None
        self.neutral_idx = None

        for idx, lbl in self.id2label.items():
            l_lower = str(lbl).lower()
            if "entail" in l_lower or l_lower == "supported" or (self.num_labels == 2 and idx == 1):
                self.entail_idx = int(idx)
            elif "contra" in l_lower or (self.num_labels == 2 and idx == 0):
                self.contra_idx = int(idx)
            elif "neutral" in l_lower:
                self.neutral_idx = int(idx)

        if self.entail_idx is None:
            self.entail_idx = 1
        if self.contra_idx is None:
            self.contra_idx = 0
        if self.neutral_idx is None and self.num_labels > 2:
            self.neutral_idx = 2

    def set_temperature(self, temp: float):
        self.temperature = max(1e-3, temp)

    def _score_single_pair(self, text_a: str, text_b: str) -> Dict[str, float]:
        """
        Passes text_a (evidence context) and text_b (claim/proposition) through cross-encoder.
        """
        if self.num_labels == 2:
            enc = self.tokenizer(
                [f"{text_a} [SEP] {text_b}"],
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            enc = self.tokenizer(
                text_a,
                text_b,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float16):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0] / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

        prob_entail = float(probs[self.entail_idx])
        prob_contra = float(probs[self.contra_idx])
        prob_neutral = float(probs[self.neutral_idx]) if self.neutral_idx is not None else 0.0

        return {
            "entailment": prob_entail,
            "contradiction": prob_contra,
            "insufficient": prob_neutral,
            "raw_probs": probs.tolist()
        }

    def verify(self, source: str, claim: str) -> Dict[str, Any]:
        """
        End-to-end multi-stage verification.
        """
        t0 = time.perf_counter_ns()

        # Step 0: Full-claim structured verification
        if self.use_structured_checks:
            full_res = StructuredClaimChecker.verify_alignment(claim, source)
            if full_res.has_hard_contradiction:
                dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
                return {
                    "decision": "NOT_SUPPORTED",
                    "probability": 0.001,
                    "internal_label": "CONTRADICTION",
                    "evidence_spans": [source[:120]],
                    "propositions": [{"proposition_id": "P1", "proposition_text": claim, "decision": "NOT_SUPPORTED", "reason": full_res.contradiction_reason}],
                    "reason": full_res.contradiction_reason or "contradicted_by_evidence",
                    "latency_us": dt_us
                }
            elif full_res.is_explicitly_supported:
                dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
                return {
                    "decision": "SUPPORTED",
                    "probability": full_res.confidence,
                    "internal_label": "ENTAILMENT",
                    "evidence_spans": [source[:120]],
                    "propositions": [{"proposition_id": "P1", "proposition_text": claim, "decision": "SUPPORTED", "reason": "explicitly_supported_by_evidence"}],
                    "reason": "all_propositions_verified",
                    "latency_us": dt_us
                }

        # Step 1: Claim Decomposition (for compound or unverified claims)
        if self.use_decomposition:
            propositions = ClaimDecomposer.decompose(claim)
        else:
            propositions = [AtomicProposition(text=claim, proposition_id="P1")]

        all_evidence_spans = []
        proposition_results = []
        overall_supported = True
        primary_rejection_reason = None
        min_entailment_prob = 1.0

        for prop in propositions:
            # Step 2: Evidence Retrieval per proposition
            spans, evidence_ctx = HybridEvidenceRetrieverV2.retrieve_evidence(
                prop.text, source, top_k=2
            )
            for s in spans:
                if s not in all_evidence_spans:
                    all_evidence_spans.append(s)

            # Step 3: Structured Semantic Checks
            struct_result = None
            if self.use_structured_checks:
                struct_result = StructuredClaimChecker.verify_alignment(prop.text, evidence_ctx)

            # Step 4: Decision Integration
            if struct_result and struct_result.has_hard_contradiction:
                p_entail = 0.001
                p_contra = 0.999
                p_insuff = 0.0
                prop_decision = "NOT_SUPPORTED"
                prop_internal_label = "CONTRADICTION"
                prop_reason = struct_result.contradiction_reason or "contradicted_by_evidence"
            elif struct_result and struct_result.is_explicitly_supported:
                p_entail = struct_result.confidence
                p_contra = 1.0 - p_entail
                p_insuff = 0.0
                prop_decision = "SUPPORTED"
                prop_internal_label = "ENTAILMENT"
                prop_reason = "entailed_by_evidence"
            else:
                # Step 4b: Fallback to Neural Cross-Encoder
                scores = self._score_single_pair(evidence_ctx, prop.text)
                p_entail = scores["entailment"]
                p_contra = scores["contradiction"]
                p_insuff = scores["insufficient"]

                if p_entail >= 0.50:
                    prop_decision = "SUPPORTED"
                    prop_internal_label = "ENTAILMENT"
                    prop_reason = "neural_entailment_verified"
                elif p_contra > p_insuff:
                    prop_decision = "NOT_SUPPORTED"
                    prop_internal_label = "CONTRADICTION"
                    prop_reason = "neural_contradiction_detected"
                else:
                    prop_decision = "NOT_SUPPORTED"
                    prop_internal_label = "INSUFFICIENT_EVIDENCE"
                    prop_reason = "insufficient_evidence_to_support"

            min_entailment_prob = min(min_entailment_prob, p_entail)

            prop_res = {
                "proposition_id": prop.proposition_id,
                "proposition_text": prop.text,
                "evidence_context": evidence_ctx,
                "decision": prop_decision,
                "internal_label": prop_internal_label,
                "probability_entailment": p_entail,
                "probability_contradiction": p_contra,
                "probability_insufficient": p_insuff,
                "reason": prop_reason
            }
            proposition_results.append(prop_res)

            # Conservative aggregation: if any proposition fails, claim fails
            if prop_decision == "NOT_SUPPORTED" and overall_supported:
                overall_supported = False
                primary_rejection_reason = f"Proposition {prop.proposition_id} failed: {prop_reason}"

        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)

        # Overall final determination
        final_decision = "SUPPORTED" if overall_supported else "NOT_SUPPORTED"
        final_reason = primary_rejection_reason if not overall_supported else "all_propositions_verified"

        # Map internal label
        if overall_supported:
            final_internal_label = "ENTAILMENT"
        elif any(p["internal_label"] == "CONTRADICTION" for p in proposition_results):
            final_internal_label = "CONTRADICTION"
        else:
            final_internal_label = "INSUFFICIENT_EVIDENCE"

        return {
            "decision": final_decision,
            "probability": float(min_entailment_prob),
            "internal_label": final_internal_label,
            "evidence_spans": all_evidence_spans,
            "propositions": proposition_results,
            "reason": final_reason,
            "latency_us": dt_us
        }
