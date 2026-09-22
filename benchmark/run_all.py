"""
Canonical entry point for ApexDecision External Benchmark Suite.
Usage:
  python -m benchmark.run_all
  python -m benchmark.run_all --suite core
  python -m benchmark.run_all --suite reasoning
  python -m benchmark.run_all --suite multilingual
  python -m benchmark.run_all --suite tools
  python -m benchmark.run_all --suite long_context
  python -m benchmark.run_all --suite external
  python -m benchmark.run_all --suite performance
  python -m benchmark.run_all --quick
"""

import os
import sys
import time
import argparse
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark.environment import set_seed, probe_environment
from benchmark.smoke_tests import SmokeTestSuite
from benchmark.core_eval import CoreEvaluator
from benchmark.reasoning_eval import ReasoningEvaluator
from benchmark.multilingual_eval import MultilingualEvaluator
from benchmark.tools_eval import ToolsEvaluator
from benchmark.long_context_eval import LongContextEvaluator
from benchmark.external_adapter import run_external_adapter_suite
from benchmark.performance_eval import PerformanceEvaluator
from benchmark.reporter import BenchmarkReporter
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="ApexDecision External Benchmark Suite")
    parser.add_argument(
        "--suite",
        choices=["all", "core", "reasoning", "multilingual", "tools", "long_context", "external", "performance"],
        default="all",
        help="Target benchmark suite to execute (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Execute in fast validation mode (capped sample limits for CI verification)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda' or 'cpu')"
    )
    args = parser.parse_args()

    # 1. Reproducibility setup
    set_seed(42)
    start_time = time.perf_counter()

    print("=" * 80)
    print(" APEXDECISION EXTERNAL BENCHMARK RUNNER")
    print(f" Target Suite: {args.suite.upper()} | Quick Mode: {args.quick}")
    print("=" * 80)

    # 2. Probe Environment & Manifest
    manifest_path = os.path.join(ROOT_DIR, "benchmark", "manifest.yaml")
    manifest_data = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = yaml.safe_load(f)

    env_info = probe_environment(ROOT_DIR)
    print(f"[Env] OS: {env_info['operating_system']['system']} | Python: {env_info['python']['version']}")
    print(f"[Env] PyTorch: {env_info['pytorch']['version']} | GPU: {env_info['gpu']['name']} ({env_info['gpu']['total_vram_mb']} MB VRAM)")
    print(f"[Env] Model: {env_info['apex_model']['architecture']} ({env_info['apex_model']['active_parameters']:,} active params)")
    print(f"[Env] Checkpoint: {env_info['apex_model']['checkpoint_path']}")

    reporter = BenchmarkReporter(ROOT_DIR, env_info, manifest_data)
    suite_results = {}
    all_failures = []
    has_fatal_error = False

    # Initialize shared model engine
    print("\n[Init] Initializing production ApexStudent engine...")
    t_init = time.perf_counter()
    engine = ApexTypedDecisionEngine(device=args.device)
    print(f"[Init] Production engine ready in {(time.perf_counter() - t_init):.2f}s.")

    # -------------------------------------------------------------------------
    # SUITE: CORE
    # -------------------------------------------------------------------------
    if args.suite in ["all", "core"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: CORE (Smoke, Grounding, Hallucination, Typed Decisions)")
        print("-" * 80)

        # Smoke Tests
        smoke = SmokeTestSuite(engine)
        smoke_res = smoke.run_all_smoke_tests()
        suite_results["smoke_production_integrity"] = {
            "sample_count": smoke_res["total_tests"],
            "correct": smoke_res["passed_tests"],
            "incorrect": smoke_res["failed_tests"],
            "accuracy": 100.0 if smoke_res["passed"] else 0.0,
            "details": smoke_res["tests"]
        }
        print(f"  [Smoke Tests] Passed: {smoke_res['passed_tests']}/{smoke_res['total_tests']}")
        reporter.write_raw_log("core", f"Smoke tests completed: {smoke_res['passed_tests']}/{smoke_res['total_tests']}")
        if not smoke_res["passed"]:
            has_fatal_error = True

        # Core Grounding & Attribution
        core = CoreEvaluator(engine)
        g_res = core.run_grounding_eval()
        suite_results["grounding_unseen_attribution"] = g_res
        all_failures.extend(g_res.get("failures", []))
        print(f"  [Grounding & Attribution] Accuracy: {g_res['accuracy']}% ({g_res['correct']}/{g_res['sample_count']}) | Fabricated Spans: {g_res['fabricated_spans']}")
        reporter.write_raw_log("core", f"Grounding accuracy: {g_res['accuracy']}%")

        # Hallucination Refusal
        h_res = core.run_hallucination_eval()
        suite_results["hallucination_refusal"] = h_res
        all_failures.extend(h_res.get("failures", []))
        print(f"  [Hallucination Refusal] Precision: {h_res['refusal_precision']}% ({h_res['correct']}/{h_res['sample_count']})")
        reporter.write_raw_log("core", f"Hallucination refusal: {h_res['refusal_precision']}%")

        # Typed Decisions Schema
        t_res = core.run_typed_decisions_eval()
        suite_results["typed_decisions_jev_contract"] = t_res
        print(f"  [Typed Decision Contract] Schema Compliance: {t_res['schema_compliance']}%")
        reporter.write_raw_log("core", f"Typed decision contract: {t_res['schema_compliance']}%")

    # -------------------------------------------------------------------------
    # SUITE: REASONING
    # -------------------------------------------------------------------------
    if args.suite in ["all", "reasoning"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: REASONING (Multi-Hop, Arithmetic DAG, Contradictions)")
        print("-" * 80)
        reasoning = ReasoningEvaluator(engine)

        # Multi-Hop Arithmetic
        m_res = reasoning.run_multihop_arithmetic_eval()
        suite_results["reasoning_multihop_arithmetic"] = m_res
        all_failures.extend(m_res.get("failures", []))
        print(f"  [Multi-Hop Arithmetic] Exact Match: {m_res['exact_match']}% ({m_res['correct']}/{m_res['sample_count']})")
        reporter.write_raw_log("reasoning", f"Multi-hop arithmetic: {m_res['exact_match']}%")

        # Contradiction & Scope
        c_res = reasoning.run_contradiction_scope_eval()
        suite_results["reasoning_contradiction_scope"] = c_res
        all_failures.extend(c_res.get("failures", []))
        print(f"  [Contradiction & Scope] Accuracy: {c_res['accuracy']}% ({c_res['correct']}/{c_res['sample_count']})")
        reporter.write_raw_log("reasoning", f"Contradiction & scope: {c_res['accuracy']}%")

    # -------------------------------------------------------------------------
    # SUITE: MULTILINGUAL
    # -------------------------------------------------------------------------
    if args.suite in ["all", "multilingual"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: MULTILINGUAL (90 Languages, 18 Script Families)")
        print("-" * 80)
        multi_path = os.path.join(ROOT_DIR, "evaluation", "multilingual_ood_v8.json")
        multilingual = MultilingualEvaluator(multi_path, agent=engine.agent)
        limit = 30 if args.quick else None
        ml_res = multilingual.run_multilingual_eval(limit=limit)
        suite_results["multilingual_fixed_ood"] = ml_res
        all_failures.extend(ml_res.get("failures", []))
        print(f"  [Multilingual OOD] Accuracy: {ml_res['accuracy']}% ({ml_res['correct']}/{ml_res['sample_count']} cases across {ml_res.get('total_languages', 0)} languages)")
        reporter.write_raw_log("multilingual", f"Multilingual OOD accuracy: {ml_res['accuracy']}%")

    # -------------------------------------------------------------------------
    # SUITE: TOOLS
    # -------------------------------------------------------------------------
    if args.suite in ["all", "tools"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: TOOLS (20 Categories Native Tool Orchestration)")
        print("-" * 80)
        tool_path = os.path.join(ROOT_DIR, "evaluation", "v8_tool_benchmark.json")
        tool_eval = ToolsEvaluator(tool_path)
        limit = 40 if args.quick else None
        t_res = tool_eval.run_tools_eval(limit=limit)
        suite_results["tools_orchestration_matrix"] = t_res
        all_failures.extend(t_res.get("failures", []))
        print(f"  [Tool Orchestration] Decision Accuracy: {t_res['decision_accuracy']}% ({t_res['correct']}/{t_res['sample_count']})")
        print(f"  [Tool Selection] Accuracy: {t_res['tool_selection_accuracy']}% | Unsafe Block Rate: {t_res['unsafe_action_prevention_rate']}% | Injection Defense: {t_res['prompt_injection_defense_rate']}%")
        reporter.write_raw_log("tools", f"Tool decision accuracy: {t_res['decision_accuracy']}%")

    # -------------------------------------------------------------------------
    # SUITE: LONG CONTEXT
    # -------------------------------------------------------------------------
    if args.suite in ["all", "long_context"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: LONG CONTEXT (32K to 1M Tokens)")
        print("-" * 80)
        lc_eval = LongContextEvaluator(engine)
        depths = ["32K", "64K", "128K"] if args.quick else ["32K", "64K", "128K", "256K", "512K", "1M"]
        lc_res = lc_eval.run_long_context_eval(depths=depths)
        suite_results["long_context_scaling"] = lc_res
        print(f"  [Long Context] Needle Retrieval Accuracy: {lc_res['retrieval_accuracy']}% across {lc_res['tested_depths']}")
        reporter.write_raw_log("long_context", f"Long context retrieval accuracy: {lc_res['retrieval_accuracy']}%")

    # -------------------------------------------------------------------------
    # SUITE: EXTERNAL ADAPTER
    # -------------------------------------------------------------------------
    if args.suite in ["all", "external"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: EXTERNAL ADAPTER (EleutherAI LM-Eval & OpenCompass)")
        print("-" * 80)
        ext_res = run_external_adapter_suite(engine)
        suite_results["external_standard_evaluation"] = ext_res
        eleuther_acc = ext_res["tasks"]["eleuther_arc_challenge"]["accuracy"]
        oc_status = ext_res["tasks"]["opencompass_classification"]["status"]
        print(f"  [EleutherAI Adapter] ARC Challenge Accuracy: {eleuther_acc}%")
        print(f"  [OpenCompass Adapter] Classification: {oc_status} | Generative: {ext_res['tasks']['opencompass_generative_unsupported']['status']}")
        reporter.write_raw_log("external", f"EleutherAI accuracy: {eleuther_acc}%")

    # -------------------------------------------------------------------------
    # SUITE: PERFORMANCE
    # -------------------------------------------------------------------------
    if args.suite in ["all", "performance"]:
        print("\n" + "-" * 80)
        print(">> Running Suite: PERFORMANCE (Latency, Throughput, VRAM)")
        print("-" * 80)
        perf_eval = PerformanceEvaluator(engine)
        test_runs = 10 if args.quick else 30
        p_res = perf_eval.run_performance_eval(warmup_runs=3, test_runs=test_runs)
        suite_results["performance_latency_profiling"] = p_res
        print(f"  [Latency] Cold: {p_res['cold_start_ms']} ms | P50: {p_res['warm_latency_ms']['p50']} ms | P95: {p_res['warm_latency_ms']['p95']} ms")
        print(f"  [Throughput] {p_res['throughput_req_per_sec']} req/s | Peak VRAM: {p_res['peak_gpu_vram_mb']} MB")
        reporter.write_raw_log("performance", f"Warm latency P50: {p_res['warm_latency_ms']['p50']} ms")

    # 3. Assemble and Write Artifacts
    print("\n" + "=" * 80)
    print(" COMPILING FINAL BENCHMARK ARTIFACTS")
    print("=" * 80)
    output_dir = reporter.compile_report(suite_results, all_failures, start_time)
    print(f"Artifacts successfully written to: {output_dir}")
    print(f"  - {os.path.join(output_dir, 'summary.json')}")
    print(f"  - {os.path.join(output_dir, 'metrics.csv')}")
    print(f"  - {os.path.join(output_dir, 'report.html')}")
    print(f"  - {os.path.join(output_dir, 'environment.json')}")
    print(f"  - {os.path.join(output_dir, 'failures.jsonl')}")

    total_time = time.perf_counter() - start_time
    print(f"\nExecution completed in {total_time:.2f}s.")

    if has_fatal_error:
        print("\n[ALERT] One or more critical production smoke tests failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
