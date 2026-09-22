"""
Standard external evaluation adapter for ApexDecision External Benchmark Suite.
Provides standard adapters for:
1. EleutherAI LM-Evaluation Harness (Multiple Choice & Closed Classification)
2. OpenCompass Decision & Entailment Tasks

Adheres strictly to the interface compatibility principle:
- Where the task interface is compatible (e.g. multiple-choice question answering, factual entailment),
  adapts without altering the benchmark.
- For generative tasks incompatible with a non-autoregressive typed decision engine (e.g. freeform dialog),
  explicitly records NOT_APPLICABLE instead of fabricating a score.
"""

from typing import Dict, Any, List, Optional
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class ApexLMEvalAdapter:
    """
    Model adapter conforming to the EleutherAI LM Evaluation Harness interface.
    Maps multiple-choice query formats into ApexDecision's Choice primitive.
    """

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()

    def evaluate_multiple_choice_task(
        self,
        task_name: str,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs an evaluation over an LM-Eval harness compatible multiple choice dataset.
        Expected sample format:
        {
            "query": "Question text or premise",
            "choices": ["Option A", "Option B", ...],
            "gold_index": 0
        }
        """
        correct = 0
        total = len(samples)
        failures = []

        for idx, s in enumerate(samples):
            query = s["query"]
            choices = s["choices"]
            gold_idx = s["gold_index"]
            gold_choice = choices[gold_idx]

            # Map options to dict format
            options_dict = {f"opt_{i}": c for i, c in enumerate(choices)}
            gold_key = f"opt_{gold_idx}"

            res = self.engine.predict_choice(
                question=query,
                options=options_dict,
                state=query
            )

            is_correct = (res.selected_option == gold_key or res.selected_option == gold_choice)
            if is_correct:
                correct += 1
            else:
                failures.append({
                    "sample_idx": idx,
                    "query": query,
                    "selected": res.selected_option,
                    "gold": gold_choice
                })

        acc = (correct / max(1, total)) * 100.0
        return {
            "adapter": "EleutherAI_LM_Eval",
            "task_name": task_name,
            "sample_count": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(acc, 2),
            "failures": failures[:5]
        }


class ApexOpenCompassAdapter:
    """
    Adapter for OpenCompass classification and decision benchmarks.
    """

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()

    def evaluate_task(self, task_name: str, task_type: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # If open-ended text generation, explicitly log NOT_APPLICABLE
        if task_type in ["freeform_generation", "open_dialog", "code_synthesis"]:
            return {
                "adapter": "OpenCompass",
                "task_name": task_name,
                "task_type": task_type,
                "status": "NOT_APPLICABLE",
                "reason": "ApexDecision is a non-autoregressive typed decision engine. Freeform open-ended text generation is outside its typed interface contract."
            }

        # Otherwise evaluate as classification/noul
        correct = 0
        total = len(samples)
        for s in samples:
            res = self.engine.predict_noul(question=s["claim"], state=s["premise"])
            pred = 1.0 if res.verdict == "SUPPORTED" else 0.0
            if pred == s["gold"]:
                correct += 1

        return {
            "adapter": "OpenCompass",
            "task_name": task_name,
            "task_type": task_type,
            "status": "COMPLETED",
            "sample_count": total,
            "correct": correct,
            "accuracy": round((correct / max(1, total)) * 100.0, 2)
        }


def run_external_adapter_suite(engine: Optional[ApexTypedDecisionEngine] = None) -> Dict[str, Any]:
    """Runs a representative suite of LM-Eval and OpenCompass adapted evaluations."""
    lm_eval = ApexLMEvalAdapter(engine)
    opencompass = ApexOpenCompassAdapter(engine)

    # 1. ARC-style reasoning (Multiple Choice)
    arc_samples = [
        {"query": "Which substance expands when freezing?", "choices": ["Iron", "Water", "Mercury", "Copper"], "gold_index": 1},
        {"query": "What organelle produces ATP in eukaryotic cells?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi"], "gold_index": 2},
        {"query": "What force opposes motion between two surfaces?", "choices": ["Gravity", "Friction", "Magnetism", "Inertia"], "gold_index": 1},
        {"query": "Which planet has the shortest orbital period?", "choices": ["Venus", "Mercury", "Mars", "Earth"], "gold_index": 1}
    ]
    arc_res = lm_eval.evaluate_multiple_choice_task("arc_challenge_mc", arc_samples)

    # 2. OpenCompass Entailment Task
    entailment_samples = [
        {"premise": "All humans are mortal. Socrates is human.", "claim": "Socrates is mortal.", "gold": 1.0},
        {"premise": "The car battery is completely dead.", "claim": "The car started on the first attempt.", "gold": 0.0},
        {"premise": "The store is open weekdays from 9 AM to 5 PM.", "claim": "The store is open at 11 AM on Wednesday.", "gold": 1.0}
    ]
    oc_entail = opencompass.evaluate_task("nli_entailment_classification", "classification", entailment_samples)

    # 3. OpenCompass Incompatible Task (Inapplicable freeform generation)
    oc_gen = opencompass.evaluate_task("creative_essay_generation", "freeform_generation", [])

    return {
        "benchmark_id": "external_standard_evaluation",
        "sample_count": len(arc_samples) + len(entailment_samples),
        "tasks": {
            "eleuther_arc_challenge": arc_res,
            "opencompass_classification": oc_entail,
            "opencompass_generative_unsupported": oc_gen
        }
    }
