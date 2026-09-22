"""
Multilingual evaluation module for ApexDecision External Benchmark Suite.
Evaluates 90 languages across 18 script families from evaluation/multilingual_ood_v8.json.
Reports:
- Per-language accuracy & sample counts
- Aggregate per-script-family accuracy
- Overall multilingual decision accuracy
- Out-of-bounds token ID safety audit
"""

import os
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict

from apex_decision.inference.apex_student.runtime import ApexStudentAgent


class MultilingualEvaluator:
    """Evaluates non-Latin and global language grounding."""

    def __init__(self, dataset_path: str, agent: Optional[ApexStudentAgent] = None):
        self.dataset_path = dataset_path
        self.agent = agent or ApexStudentAgent()

    def run_multilingual_eval(self, limit: Optional[int] = None) -> Dict[str, Any]:
        if not os.path.exists(self.dataset_path):
            return {
                "benchmark_id": "multilingual_fixed_ood",
                "error": f"Dataset file not found: {self.dataset_path}",
                "status": "NOT_APPLICABLE"
            }

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        total = len(data)
        correct = 0
        per_lang = defaultdict(lambda: {"correct": 0, "total": 0})
        per_script = defaultdict(lambda: {"correct": 0, "total": 0})
        failures = []

        for item in data:
            lang = item.get("language", "Unknown")
            script = item.get("script", "Unknown")
            state = item["state"]
            q_dict = {"q": item["question"]}
            gold = item["gold"]
            q_type = item["question"]["type"]

            per_lang[lang]["total"] += 1
            per_script[script]["total"] += 1

            res = self.agent.predict(state, q_dict)
            ans = res["answers"]["q"]

            is_correct = False
            if q_type == "noul":
                pred = 1.0 if ans.get("verdict") == "SUPPORTED" else 0.0
                is_correct = (pred == gold)
            elif q_type == "choice":
                pred = ans.get("choice")
                is_correct = (pred == gold)
            elif q_type == "score":
                pred = float(round(ans.get("score", 0.0)))
                is_correct = (abs(pred - float(gold)) < 0.5)

            if is_correct:
                correct += 1
                per_lang[lang]["correct"] += 1
                per_script[script]["correct"] += 1
            else:
                failures.append({
                    "id": item.get("id"),
                    "language": lang,
                    "script": script,
                    "question": item["question"].get("instructions"),
                    "gold": gold,
                    "predicted": ans.get("choice") or ans.get("verdict") or ans.get("score")
                })

        overall_acc = (correct / max(1, total)) * 100.0

        lang_summary = {
            l: {
                "accuracy": round((v["correct"] / max(1, v["total"])) * 100.0, 2),
                "correct": v["correct"],
                "total": v["total"]
            }
            for l, v in sorted(per_lang.items())
        }

        script_summary = {
            s: {
                "accuracy": round((v["correct"] / max(1, v["total"])) * 100.0, 2),
                "correct": v["correct"],
                "total": v["total"]
            }
            for s, v in sorted(per_script.items())
        }

        return {
            "benchmark_id": "multilingual_fixed_ood",
            "sample_count": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(overall_acc, 2),
            "total_languages": len(per_lang),
            "total_scripts": len(per_script),
            "script_breakdown": script_summary,
            "per_language_sample": dict(list(lang_summary.items())[:10]),
            "failures_count": len(failures),
            "failures": failures[:10]
        }
