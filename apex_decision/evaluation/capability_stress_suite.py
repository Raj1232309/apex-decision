"""
ApexDecision V7.2: Comprehensive Capability-Parity Stress Suite.
Empirically evaluates Laya (ModernBERT-large 421M) vs ApexStudent (19.5M from-scratch)
across 20+ distinct decision, semantic, scaling, multilingual, and reliability capabilities
under strictly matched conditions on local CUDA hardware (NVIDIA GeForce RTX 2050).
"""

import os
import sys
import time
import json
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

# Add paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "laya_repo"))

import laya
from apex_decision.models.apex_student.config import ApexStudentConfig
from apex_decision.models.apex_student.model import ApexStudentModel
from apex_decision.models.apex_student.tokenizer import ApexTokenizer
from apex_decision.inference.apex_student.runtime import ApexStudentAgent
from apex_decision.verification.provenance_verifier import ProvenanceVerifier


def build_capability_test_battery() -> List[Dict[str, Any]]:
    """
    Constructs a diverse, non-repetitive battery of test cases covering all 20+ capabilities.
    """
    battery = []

    # ---------------------------------------------------------
    # 1. DECISION PRIMITIVES: NOUL (Binary / Nullary Truth)
    # ---------------------------------------------------------
    battery.append({
        "id": "noul_supported_01",
        "category": "decision:noul",
        "state": "The audit confirms operating cash flow reached $412M in fiscal 2025.",
        "question": {"type": "noul", "instructions": "Did operating cash flow reach $412M in fiscal 2025?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Standard binary factual assertion"
    })
    battery.append({
        "id": "noul_unsupported_01",
        "category": "decision:noul",
        "state": "Operating cash flow reached $412M in fiscal 2025, but capital expenditure was not disclosed.",
        "question": {"type": "noul", "instructions": "Did capital expenditure reach $500M in fiscal 2025?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Unsupported claim with adjacent factual context"
    })

    # ---------------------------------------------------------
    # 2. DECISION PRIMITIVES: CHOICE (Multi-Class Categorical)
    # ---------------------------------------------------------
    battery.append({
        "id": "choice_triage_01",
        "category": "decision:choice",
        "state": "User reports account locked due to five failed MFA attempts during morning shift.",
        "question": {
            "type": "choice",
            "instructions": "Select the appropriate support triage department queue.",
            "criteria": {
                "billing_support": "Payments, invoices, and refunds",
                "identity_security": "MFA, login locks, authentication failures",
                "infrastructure_ops": "Server downtime, network latency",
                "general_inquiry": "Non-technical product questions"
            }
        },
        "gold": "identity_security",
        "is_adversarial": False,
        "description": "Categorical routing based on specific semantic trigger"
    })
    battery.append({
        "id": "choice_fine_grained_01",
        "category": "decision:choice",
        "state": "Customer states: 'The item arrived shattered in transit; please issue a refund to my Visa.'",
        "question": {
            "type": "choice",
            "instructions": "Determine customer intent.",
            "criteria": {
                "request_exchange": "Customer wants a replacement unit",
                "request_refund": "Customer wants monetary refund for damaged item",
                "shipping_status": "Tracking delayed package",
                "cancel_order": "Canceling unshipped order"
            }
        },
        "gold": "request_refund",
        "is_adversarial": True,
        "description": "Adversarial near-miss choice between exchange vs refund"
    })

    # ---------------------------------------------------------
    # 3. DECISION PRIMITIVES: SCORE (Continuous Ordinal Scale)
    # ---------------------------------------------------------
    battery.append({
        "id": "score_severity_01",
        "category": "decision:score",
        "state": "Critical data exfiltration confirmed on primary production database cluster with active root compromise.",
        "question": {
            "type": "score",
            "instructions": "Rate incident severity.",
            "criteria": ["low: informational notice", "medium: isolated non-critical service", "high: customer impacting", "critical: full breach of primary infrastructure"]
        },
        "gold": 3, # Level 3: critical
        "is_adversarial": False,
        "description": "Continuous score mapping to highest severity level"
    })
    battery.append({
        "id": "score_sentiment_01",
        "category": "decision:score",
        "state": "The delivery was acceptable, arrived on time without damages, though the packaging was basic.",
        "question": {
            "type": "score",
            "instructions": "Rate customer satisfaction on a 3-point scale.",
            "criteria": ["dissatisfied", "neutral", "satisfied"]
        },
        "gold": 1, # Level 1: neutral/acceptable
        "is_adversarial": True,
        "description": "Mild neutral sentiment evaluation on ordinal scale"
    })

    # ---------------------------------------------------------
    # 4. DECISION PRIMITIVES: VERIFY & ABSTAIN
    # ---------------------------------------------------------
    battery.append({
        "id": "verify_abstain_01",
        "category": "decision:abstain",
        "state": "The user inquiry asks about the proprietary battery chemistry of the upcoming 2027 prototype.",
        "question": {"type": "noul", "instructions": "Does the document disclose the exact cathode ratio for 2027?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Abstain / reject ungrounded proprietary disclosure"
    })

    # ---------------------------------------------------------
    # 5. SEMANTIC: ENTAILMENT & CONTRADICTION
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_entail_01",
        "category": "semantic:entailment",
        "state": "Acme Corp finalized the acquisition of BetaSoft for $450M in all-cash consideration.",
        "question": {"type": "noul", "instructions": "Did Acme Corp acquire BetaSoft?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Direct semantic entailment"
    })
    battery.append({
        "id": "sem_contradiction_01",
        "category": "semantic:contradiction",
        "state": "Acme Corp terminated negotiations and rejected the proposed acquisition of BetaSoft.",
        "question": {"type": "noul", "instructions": "Did Acme Corp acquire BetaSoft?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Explicit negation of acquisition transaction"
    })

    # ---------------------------------------------------------
    # 6. SEMANTIC: NEGATION POLARITY
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_negation_01",
        "category": "semantic:negation",
        "state": "The clinical trial demonstrated no statistically significant reduction in patient mortality rates.",
        "question": {"type": "noul", "instructions": "Did the trial demonstrate a significant reduction in mortality?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Syntactic negation 'no statistically significant reduction'"
    })
    battery.append({
        "id": "sem_negation_double_01",
        "category": "semantic:negation",
        "state": "It is not inaccurate to state that operating margins expanded during the second quarter.",
        "question": {"type": "noul", "instructions": "Did operating margins expand during the second quarter?"},
        "gold": 1.0,
        "is_adversarial": True,
        "description": "Double negation 'not inaccurate to state that margins expanded'"
    })

    # ---------------------------------------------------------
    # 7. SEMANTIC: PARAPHRASE & SYNONYMY
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_paraphrase_01",
        "category": "semantic:paraphrase",
        "state": "The firm witnessed an unprecedented surge in subscription renewals across European markets.",
        "question": {"type": "noul", "instructions": "Did customer retention increase significantly in Europe?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Lexical paraphrase (subscription renewals surge -> retention increase)"
    })
    battery.append({
        "id": "sem_synonym_01",
        "category": "semantic:synonymy",
        "state": "The executive board consented to the divestiture of the retail hardware division.",
        "question": {"type": "noul", "instructions": "Did leadership approve selling the retail hardware business?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Synonym mapping (consented -> approved, divestiture -> selling)"
    })

    # ---------------------------------------------------------
    # 8. SEMANTIC: ACRONYMS & ENTITY RESOLUTION
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_acronym_01",
        "category": "semantic:acronyms",
        "state": "The company reported ARR of $120M while maintaining GAAP operating margin of 14%.",
        "question": {"type": "noul", "instructions": "Did annualized recurring revenue reach $120 million?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Acronym expansion (ARR -> annualized recurring revenue)"
    })
    battery.append({
        "id": "sem_entity_01",
        "category": "semantic:entity_resolution",
        "state": "Big Blue unveiled its newest 1,000-qubit quantum processor in Yorktown Heights.",
        "question": {"type": "noul", "instructions": "Did IBM announce a quantum processor?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Entity resolution (Big Blue -> IBM)"
    })

    # ---------------------------------------------------------
    # 9. SEMANTIC: TEMPORAL REASONING
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_temporal_before_01",
        "category": "semantic:temporal",
        "state": "The merger agreement was executed in March 2023, prior to the regulatory antitrust review in November 2023.",
        "question": {"type": "noul", "instructions": "Was the merger executed before the antitrust review?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Temporal ordering 'prior to'"
    })
    battery.append({
        "id": "sem_temporal_adv_01",
        "category": "semantic:temporal",
        "state": "The merger agreement was executed in March 2023, prior to the regulatory antitrust review in November 2023.",
        "question": {"type": "noul", "instructions": "Was the regulatory antitrust review conducted before the merger was executed?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Adversarial inverted temporal order"
    })

    # ---------------------------------------------------------
    # 10. SEMANTIC: NUMERIC REASONING, PERCENTAGES & INEQUALITIES
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_numeric_exact_01",
        "category": "semantic:numeric",
        "state": "Total active subscribers reached 48.3 million by end of year.",
        "question": {"type": "noul", "instructions": "Did subscriber count reach 48.3 million?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Exact numeric verification"
    })
    battery.append({
        "id": "sem_numeric_near_miss_01",
        "category": "semantic:numeric",
        "state": "Total active subscribers reached 48.3 million by end of year.",
        "question": {"type": "noul", "instructions": "Did subscriber count reach 48.8 million?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Adversarial near-miss numeric perturbation (48.3M vs 48.8M)"
    })
    battery.append({
        "id": "sem_percentage_01",
        "category": "semantic:percentages",
        "state": "Gross profit margin contracted by 180 basis points to 64.2%.",
        "question": {"type": "noul", "instructions": "Did gross profit margin reach 64.2%?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Percentage margin extraction"
    })
    battery.append({
        "id": "sem_inequality_01",
        "category": "semantic:inequalities",
        "state": "The contract penalty fee is $75,000 for each month of delayed delivery.",
        "question": {"type": "noul", "instructions": "Does the monthly delay penalty exceed $50,000?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Inequality greater-than reasoning ($75,000 > $50,000)"
    })
    battery.append({
        "id": "sem_inequality_adv_01",
        "category": "semantic:inequalities",
        "state": "The contract penalty fee is $75,000 for each month of delayed delivery.",
        "question": {"type": "noul", "instructions": "Is the monthly delay penalty less than $50,000?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Adversarial inequality contradiction"
    })

    # ---------------------------------------------------------
    # 11. SEMANTIC: MULTI-HOP REASONING & DISTRACTORS
    # ---------------------------------------------------------
    battery.append({
        "id": "sem_multihop_01",
        "category": "semantic:multihop",
        "state": "Solaris Corp is the parent company of Lumina Labs. Lumina Labs owns the patented photovoltaic cell technology used in Project Aurora.",
        "question": {"type": "noul", "instructions": "Does Solaris Corp's subsidiary hold the patent for Project Aurora's photovoltaic cell?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "2-hop entity and ownership link"
    })
    battery.append({
        "id": "sem_distractor_01",
        "category": "semantic:distractor_resistance",
        "state": "Competitor Alpha reported revenues of $900M. In contrast, our company closed the year at $310M due to supply chain headwinds.",
        "question": {"type": "noul", "instructions": "Did our company achieve $900M in revenue?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Adjacent distractor entity with competing numeric figure"
    })
    battery.append({
        "id": "sem_ambiguous_01",
        "category": "semantic:ambiguous_inputs",
        "state": "Certain operations were paused during the quarter due to weather events.",
        "question": {"type": "noul", "instructions": "Were European operations specifically halted due to severe blizzards?"},
        "gold": 0.0,
        "is_adversarial": True,
        "description": "Underspecified regional scope rejecting ungrounded specificity"
    })

    # ---------------------------------------------------------
    # 12. MULTILINGUAL & SCRIPT HANDLING
    # ---------------------------------------------------------
    battery.append({
        "id": "multi_german_01",
        "category": "multilingual:latin_extended",
        "state": "Der Reingewinn des Unternehmens stieg im vierten Quartal um 12 Prozent auf 85 Millionen Euro.",
        "question": {"type": "noul", "instructions": "Stieg der Reingewinn um 12 Prozent?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "German language financial assertion"
    })
    battery.append({
        "id": "multi_french_01",
        "category": "multilingual:latin_extended",
        "state": "Le chiffre d'affaires annuel a atteint 150 millions d'euros avec une marge stable.",
        "question": {"type": "noul", "instructions": "Le chiffre d'affaires a-t-il atteint 150 millions d'euros?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "French language revenue assertion"
    })
    battery.append({
        "id": "multi_russian_01",
        "category": "multilingual:cyrillic",
        "state": "Выручка компании за третий квартал составила 45 миллиардов рублей.",
        "question": {"type": "noul", "instructions": "Составила ли выручка 45 миллиардов рублей?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Russian Cyrillic script assertion"
    })
    battery.append({
        "id": "multi_hindi_01",
        "category": "multilingual:devanagari",
        "state": "कंपनी का शुद्ध लाभ पिछले वर्ष की तुलना में 15 प्रतिशत बढ़ गया।",
        "question": {"type": "noul", "instructions": "क्या शुद्ध लाभ 15 प्रतिशत बढ़ गया?"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Hindi Devanagari script assertion"
    })
    battery.append({
        "id": "multi_chinese_01",
        "category": "multilingual:cjk",
        "state": "公司第四季度总收入达到五十亿元人民币，同比增长百分之八。",
        "question": {"type": "noul", "instructions": "第四季度总收入是否达到五十亿元？"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Simplified Chinese CJK assertion"
    })
    battery.append({
        "id": "multi_arabic_01",
        "category": "multilingual:arabic",
        "state": "أعلنت الشركة عن تحقيق أرباح صافية قدرها 20 مليون دولار خلال الربع الأول.",
        "question": {"type": "noul", "instructions": "هل بلغت الأرباح الصافية 20 مليون دولار؟"},
        "gold": 1.0,
        "is_adversarial": False,
        "description": "Arabic script assertion"
    })

    return battery


def evaluate_choice_scaling(student_agent, laya_agent, cardinalities: List[int] = [2, 5, 10, 20, 50, 77, 100, 255]) -> Dict[str, Any]:
    scaling_results = {}
    for k in cardinalities:
        opts = [f"opt_queue_routing_code_{i:03d}" for i in range(k)]
        target_idx = min(7, k - 1)
        target = opts[target_idx]
        state = f"Customer request indicates immediate dispatch to {target} under corporate service agreement."
        q = {
            "q_scale": {
                "type": "choice",
                "instructions": "Select the correct destination queue code.",
                "criteria": {o: None for o in opts}
            }
        }

        # Student evaluation
        student_correct = False
        student_pred = None
        student_invented = False
        try:
            s_res = student_agent.predict(state, q)
            student_pred = s_res["answers"]["q_scale"]["choice"]
            student_correct = (student_pred == target)
            student_invented = student_pred not in opts
        except Exception as e:
            student_pred = f"ERROR: {e}"

        # Laya evaluation
        laya_correct = False
        laya_pred = None
        laya_invented = False
        laya_err = None
        try:
            # Laya has head_max_len limitation; might fail or truncate
            l_res = laya_agent.predict(state, q)
            laya_pred = l_res["answers"]["q_scale"]["choice"]
            laya_correct = (laya_pred == target)
            laya_invented = laya_pred not in opts
        except Exception as e:
            laya_err = str(e)
            laya_pred = f"ERROR: {e}"

        scaling_results[str(k)] = {
            "options": k,
            "target": target,
            "student_pred": student_pred,
            "student_correct": student_correct,
            "student_invented": student_invented,
            "laya_pred": laya_pred,
            "laya_correct": laya_correct,
            "laya_invented": laya_invented,
            "laya_error": laya_err
        }
        print(f"  [Scaling k={k:3d}] Student: {'PASS' if student_correct else 'FAIL'} ({student_pred}) | Laya: {'PASS' if laya_correct else 'FAIL'} ({laya_pred})", flush=True)

    return scaling_results


def run_capability_stress_suite():
    print("=" * 80)
    print("APEXDECISION V7.2: COMPREHENSIVE CAPABILITY-PARITY AUDIT SUITE")
    print("Evaluating Laya (ModernBERT-large 421M) vs ApexStudent (From-Scratch 19.5M)")
    print("Device: NVIDIA GeForce RTX 2050 (CUDA) | Fixed Byte-Identical Inputs")
    print("=" * 80)

    # 1. Load ApexStudent
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ApexTokenizer.from_pretrained("apex_decision/models/apex_student/tokenizer_weights")
    cfg = ApexStudentConfig(vocab_size=max(4096, tok.vocab_size), hidden_size=384, num_hidden_layers=8, num_attention_heads=8)
    student_model = ApexStudentModel(cfg).to(dev)
    ckpt_path = "apex_decision/models/apex_student/checkpoints/best_student.pt"
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=dev)
        student_model.load_state_dict(sd.get("model_state", sd), strict=False)
        print("[Load] ApexStudent checkpoint loaded successfully!")
    else:
        print("[Load] WARNING: ApexStudent checkpoint not found!")

    student_agent = ApexStudentAgent(student_model, tok, cfg, device=dev)

    # 2. Load Laya
    print("[Load] Loading Laya baseline agent from laya_weights...")
    laya_agent = laya.load("laya_weights", device=dev)
    print(f"[Load] Laya baseline loaded successfully on {laya_agent.device}!")

    # 3. Run Test Battery
    battery = build_capability_test_battery()
    print(f"\n[Run] Executing battery of {len(battery)} tests across 20+ capabilities...")

    results = []
    category_summary = {}

    student_total_lat = 0.0
    laya_total_lat = 0.0

    for idx, test in enumerate(battery):
        cid = test["id"]
        cat = test["category"]
        state = test["state"]
        q_dict = {"q": test["question"]}
        gold = test["gold"]
        q_type = test["question"]["type"]

        # Run Student
        t0 = time.perf_counter()
        s_res = student_agent.predict(state, q_dict)
        s_lat = (time.perf_counter() - t0) * 1000.0
        student_total_lat += s_lat

        s_ans = s_res["answers"]["q"]
        s_pred = None
        s_conf = s_ans.get("confidence", 0.5)
        if q_type == "noul":
            s_val = s_ans.get("noul", 0.5)
            s_verdict = s_ans.get("verdict", "SUPPORTED" if s_val >= 0.5 else "NOT_SUPPORTED")
            s_pred = 1.0 if s_verdict == "SUPPORTED" else 0.0
            s_correct = (s_pred == gold)
        elif q_type == "choice":
            s_pred = s_ans.get("choice")
            s_correct = (s_pred == gold)
        elif q_type == "score":
            s_pred = int(round(s_ans.get("score", 0.0)))
            s_correct = (s_pred == gold)

        # Run Laya
        t0 = time.perf_counter()
        l_pred = None
        l_conf = 0.5
        l_correct = False
        l_err = None
        try:
            l_res = laya_agent.predict(state, q_dict)
            l_lat = (time.perf_counter() - t0) * 1000.0
            laya_total_lat += l_lat
            l_ans = l_res["answers"]["q"]
            l_conf = l_ans.get("confidence", 0.5)

            if q_type == "noul":
                l_val = l_ans.get("noul", 0.5)
                l_pred = 1.0 if l_val >= 0.5 else 0.0
                l_correct = (l_pred == gold)
            elif q_type == "choice":
                l_pred = l_ans.get("choice")
                l_correct = (l_pred == gold)
            elif q_type == "score":
                l_pred = int(round(l_ans.get("score", 0.0)))
                l_correct = (l_pred == gold)
        except Exception as e:
            l_lat = (time.perf_counter() - t0) * 1000.0
            l_err = str(e)
            l_pred = f"ERROR: {e}"

        item_res = {
            "id": cid,
            "category": cat,
            "is_adversarial": test["is_adversarial"],
            "student": {
                "pred": s_pred,
                "confidence": s_conf,
                "correct": s_correct,
                "latency_ms": round(s_lat, 2)
            },
            "laya": {
                "pred": l_pred,
                "confidence": l_conf,
                "correct": l_correct,
                "latency_ms": round(l_lat, 2),
                "error": l_err
            }
        }
        results.append(item_res)
        print(f"[{idx+1:02d}/{len(battery)}] {cid:26} | Student: {'PASS' if s_correct else 'FAIL'} | Laya: {'PASS' if l_correct else 'FAIL'}", flush=True)

        # Category rollups
        if cat not in category_summary:
            category_summary[cat] = {
                "total": 0,
                "student_correct": 0,
                "laya_correct": 0,
                "adversarial_total": 0,
                "student_adv_correct": 0,
                "laya_adv_correct": 0
            }
        cs = category_summary[cat]
        cs["total"] += 1
        if s_correct:
            cs["student_correct"] += 1
        if l_correct:
            cs["laya_correct"] += 1
        if test["is_adversarial"]:
            cs["adversarial_total"] += 1
            if s_correct:
                cs["student_adv_correct"] += 1
            if l_correct:
                cs["laya_adv_correct"] += 1

    # 4. Choice Scaling Benchmark (2 to 255 options)
    print("\n[Run] Benchmarking High-Cardinality Choice Scaling (2 to 255 options)...")
    scaling_res = evaluate_choice_scaling(student_agent, laya_agent)

    # 5. Peak Memory & Latency Profiling
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _ = student_agent.predict("Quick test state.", {"q": {"type": "noul", "instructions": "Is this a test?"}})
    torch.cuda.synchronize()
    student_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    torch.cuda.reset_peak_memory_stats()
    _ = laya_agent.predict("Quick test state.", {"q": {"type": "noul", "instructions": "Is this a test?"}})
    torch.cuda.synchronize()
    laya_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Compute aggregate stats
    total_tests = len(results)
    s_total_correct = sum(1 for r in results if r["student"]["correct"])
    l_total_correct = sum(1 for r in results if r["laya"]["correct"])
    s_acc = (s_total_correct / total_tests) * 100.0
    l_acc = (l_total_correct / total_tests) * 100.0

    adv_tests = [r for r in results if r["is_adversarial"]]
    s_adv_correct = sum(1 for r in adv_tests if r["student"]["correct"])
    l_adv_correct = sum(1 for r in adv_tests if r["laya"]["correct"])
    s_adv_acc = (s_adv_correct / len(adv_tests)) * 100.0
    l_adv_acc = (l_adv_correct / len(adv_tests)) * 100.0

    summary = {
        "hardware": "NVIDIA GeForce RTX 2050 (4GB)",
        "total_test_cases": total_tests,
        "adversarial_test_cases": len(adv_tests),
        "overall_student_accuracy": round(s_acc, 2),
        "overall_laya_accuracy": round(l_acc, 2),
        "adversarial_student_accuracy": round(s_adv_acc, 2),
        "adversarial_laya_accuracy": round(l_adv_acc, 2),
        "student_avg_latency_ms": round(student_total_lat / total_tests, 2),
        "laya_avg_latency_ms": round(laya_total_lat / total_tests, 2),
        "student_peak_vram_mb": round(student_peak_vram, 1),
        "laya_peak_vram_mb": round(laya_peak_vram, 1),
        "category_summary": category_summary,
        "choice_scaling": scaling_res,
        "detailed_results": results
    }

    out_file = "apex_decision/reports/capability_stress_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Audit Results Saved] -> {out_file}")
    print("\n--- OVERALL COMPARISON SCOREBOARD ---")
    print(f"Student Overall Accuracy:     {s_acc:.2f}% | Laya: {l_acc:.2f}%")
    print(f"Student Adversarial Accuracy: {s_adv_acc:.2f}% | Laya: {l_adv_acc:.2f}%")
    print(f"Student Mean Latency:         {summary['student_avg_latency_ms']} ms | Laya: {summary['laya_avg_latency_ms']} ms")
    print(f"Student Peak VRAM:            {summary['student_peak_vram_mb']} MB | Laya: {summary['laya_peak_vram_mb']} MB")

    return summary


if __name__ == "__main__":
    run_capability_stress_suite()
