"""
Teacher Distillation Target Generator for ApexDecision V7.
Uses Laya as the teacher to generate calibrated soft probability targets,
marker logits, and confidence metrics over our task corpus.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from ...models.apex_student.tokenizer import ApexTokenizer


def generate_teacher_supervision(
    teacher_agent,
    corpus_items: List[Dict[str, Any]],
    output_cache_path: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Runs Laya teacher over corpus_items and records soft distributions,
    option-level probabilities, and calibrated confidence targets.

    corpus_items format:
      [
        {
          "id": "...",
          "state": "...",
          "question": {"t": "noul|choice|score", "ins": "...", "crit": {...}},
          "hard_target": [0.0, 1.0],
          "qtype": 2
        }
      ]
    """
    print(f"[Distill] Generating teacher supervision for {len(corpus_items)} items...")
    teacher_targets = []
    t0 = time.perf_counter()

    for idx, item in enumerate(corpus_items):
        state = item["state"]
        q_def = item["question"]
        qid = item.get("id", f"item_{idx}")

        questions_dict = {
            qid: {
                "type": q_def["t"],
                "instructions": q_def["ins"],
                "criteria": q_def.get("crit")
            }
        }

        try:
            # Predict using Laya teacher
            res = teacher_agent.predict(state, questions_dict)
            ans = res["answers"][qid]

            probs = ans.get("probabilities", {})
            conf = ans.get("confidence", 0.5)

            # Convert probabilities to array aligned with options
            if q_def["t"] == "choice":
                opts = list(q_def.get("crit", {}).keys())
                p_arr = [probs.get(opt, 1.0 / max(1, len(opts))) for opt in opts]
            elif q_def["t"] == "score":
                crit_len = len(q_def.get("crit", []))
                p_arr = [probs.get(str(i), 1.0 / max(1, crit_len)) for i in range(crit_len)]
            else:  # noul
                p_true = ans.get("noul", 0.5)
                p_arr = [1.0 - p_true, p_true]

            teacher_targets.append({
                "id": qid,
                "teacher_probs": p_arr,
                "teacher_conf": conf,
                "hard_target": item.get("hard_target", p_arr),
                "qtype": item["qtype"],
                "state": state,
                "question": q_def
            })
        except Exception as e:
            # Fallback to hard label if teacher execution hits an edge case
            p_fallback = item.get("hard_target", [0.5, 0.5])
            teacher_targets.append({
                "id": qid,
                "teacher_probs": p_fallback,
                "teacher_conf": 0.8,
                "hard_target": p_fallback,
                "qtype": item["qtype"],
                "state": state,
                "question": q_def
            })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(corpus_items)} items in {time.perf_counter()-t0:.1f}s")

    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    torch.save(teacher_targets, output_cache_path)
    print(f"[Distill] Successfully saved {len(teacher_targets)} targets to {output_cache_path}")
    return {"total_items": len(teacher_targets), "cache_path": output_cache_path}
