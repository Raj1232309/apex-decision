"""
ApexDecision External Benchmark Center - Gradio Web Application.
Interactive UI for running, monitoring, and downloading external benchmark evaluations.
Runs on port 7860 (Hugging Face Spaces compatible).
"""

import os
import sys
import time
import glob
import json
import threading
import subprocess
from typing import Generator, Tuple, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gradio as gr
from benchmark.environment import probe_environment

# Global execution concurrency lock
_BENCHMARK_LOCK = threading.Lock()
_CURRENT_RUN_INFO = {
    "is_running": False,
    "current_suite": None,
    "start_time": 0.0
}


def get_latest_results_dir() -> Optional[str]:
    """Finds the most recent results directory."""
    dirs = sorted(glob.glob(os.path.join(ROOT_DIR, "results", "*")))
    return dirs[-1] if dirs else None


def execute_suite_stream(suite_name: str, quick_mode: bool = False) -> Generator[Tuple[str, str, str, str, str, Optional[str], Optional[str], Optional[str]], None, None]:
    """
    Executes 'python -m benchmark.run_all --suite <suite>' in a subprocess,
    streaming logs, benchmark status, and final download files to the Gradio interface.
    """
    global _CURRENT_RUN_INFO

    if not _BENCHMARK_LOCK.acquire(blocking=False):
        yield (
            "⚠️ A benchmark run is already in progress. Concurrent execution is locked to preserve measurement integrity.",
            "LOCKED",
            "N/A",
            "N/A",
            "N/A",
            None, None, None
        )
        return

    _CURRENT_RUN_INFO["is_running"] = True
    _CURRENT_RUN_INFO["current_suite"] = suite_name
    _CURRENT_RUN_INFO["start_time"] = time.perf_counter()

    cmd = [sys.executable, "-u", "-m", "benchmark.run_all", "--suite", suite_name]
    if quick_mode:
        cmd.append("--quick")

    log_buffer = []
    current_bench = "Initializing Suite..."
    progress_status = "0/0"
    summary_markdown = "Evaluation in progress..."

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            log_buffer.append(line)
            # Parse key benchmark lines for live status
            if ">> Running Suite:" in line:
                current_bench = line.replace(">> Running Suite:", "").strip()
            elif "[" in line and "]" in line and ("Accuracy" in line or "Passed" in line or "Precision" in line):
                current_bench = line.strip()

            elapsed = time.perf_counter() - _CURRENT_RUN_INFO["start_time"]
            log_display = "".join(log_buffer[-30:])  # Show recent 30 lines
            elapsed_str = f"{elapsed:.1f}s"

            yield (
                log_display,
                current_bench,
                elapsed_str,
                "In Progress",
                "Running...",
                None, None, None
            )

        proc.wait()
        exit_code = proc.returncode
        total_time = time.perf_counter() - _CURRENT_RUN_INFO["start_time"]

        # Locate generated results
        latest_dir = get_latest_results_dir()
        summary_file = None
        metrics_file = None
        report_file = None

        if latest_dir and os.path.exists(latest_dir):
            s_candidate = os.path.join(latest_dir, "summary.json")
            m_candidate = os.path.join(latest_dir, "metrics.csv")
            r_candidate = os.path.join(latest_dir, "report.html")
            if os.path.exists(s_candidate):
                summary_file = s_candidate
            if os.path.exists(m_candidate):
                metrics_file = m_candidate
            if os.path.exists(r_candidate):
                report_file = r_candidate

        if summary_file and os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                summary_markdown = (
                    f"### Aggregate Evaluation Results\n"
                    f"- **Status:** {s_data.get('evaluation_status', 'COMPLETE')} (Exit Code: {exit_code})\n"
                    f"- **Aggregate Accuracy:** **{s_data.get('overall_accuracy', 0.0):.2f}%** "
                    f"*(95% CI: [{s_data.get('overall_ci_95', [0, 0])[0]}%, {s_data.get('overall_ci_95', [0, 0])[1]}%])*\n"
                    f"- **Total Evaluated Cases:** {s_data.get('total_samples', 0):,}\n"
                    f"- **Elapsed Execution Time:** {total_time:.2f} seconds\n"
                    f"- **Artifacts Directory:** `{latest_dir}`\n"
                )
            except Exception:
                summary_markdown = f"Benchmark completed with exit code {exit_code} in {total_time:.1f}s."
        else:
            summary_markdown = f"Benchmark completed with exit code {exit_code} in {total_time:.1f}s."

        final_logs = "".join(log_buffer)
        status_label = "COMPLETED (Success)" if exit_code == 0 else f"ERROR (Exit Code {exit_code})"

        yield (
            final_logs,
            "Evaluation Complete",
            f"{total_time:.1f}s",
            status_label,
            summary_markdown,
            summary_file,
            metrics_file,
            report_file
        )

    finally:
        _CURRENT_RUN_INFO["is_running"] = False
        _BENCHMARK_LOCK.release()


def build_app():
    env = probe_environment(ROOT_DIR)

    gpu_desc = f"{env['gpu']['name']} ({env['gpu']['total_vram_mb']} MB VRAM)" if env['gpu']['device_count'] > 0 else "CPU Mode"
    model_desc = f"{env['apex_model']['architecture']} ({env['apex_model']['active_parameters'] / 1e6:.2f}M active parameters, from scratch)"

    with gr.Blocks(title="ApexDecision External Benchmark Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # APEXDECISION EXTERNAL BENCHMARK CENTER
            **Independent Reproducible Evaluation Suite for ApexStudent-50M Production Runtime**
            
            *Hardware:* `{gpu_desc}` | *PyTorch:* `{env['pytorch']['version']}` | *Model:* `{model_desc}`
            """
        )

        with gr.Row():
            btn_all = gr.Button("🚀 Run All Benchmarks", variant="primary", scale=2)
            chk_quick = gr.Checkbox(label="Quick CI Mode (Capped)", value=False, scale=1)

        with gr.Row():
            btn_core = gr.Button("Run Core (Grounding & Smoke)")
            btn_reasoning = gr.Button("Run Reasoning (Multi-Hop & Logic)")
            btn_multilingual = gr.Button("Run Multilingual (90 Langs)")
            btn_tools = gr.Button("Run Tools (20 Categories)")

        with gr.Row():
            btn_long = gr.Button("Run Long Context (32K-1M)")
            btn_external = gr.Button("Run External (LM-Eval/OpenCompass)")
            btn_performance = gr.Button("Run Performance (Latency/VRAM)")

        gr.Markdown("### Live Execution Telemetry")
        with gr.Row():
            txt_status = gr.Textbox(label="Status", value="IDLE", interactive=False, scale=1)
            txt_current = gr.Textbox(label="Current Benchmark", value="None", interactive=False, scale=2)
            txt_time = gr.Textbox(label="Elapsed Time", value="0.0s", interactive=False, scale=1)

        txt_logs = gr.Textbox(
            label="Live Console Logs (Streaming directly from python -m benchmark.run_all)",
            lines=14,
            max_lines=25,
            interactive=False,
            autoscroll=True
        )

        out_summary = gr.Markdown(label="Aggregate Results", value="Run a benchmark suite to view aggregate scores.")

        gr.Markdown("### Download Output Artifacts")
        with gr.Row():
            file_summary = gr.File(label="summary.json", interactive=False)
            file_metrics = gr.File(label="metrics.csv", interactive=False)
            file_report = gr.File(label="report.html", interactive=False)

        # Wire Button Actions
        out_targets = [txt_logs, txt_current, txt_time, txt_status, out_summary, file_summary, file_metrics, file_report]

        btn_all.click(fn=lambda q: execute_suite_stream("all", q), inputs=[chk_quick], outputs=out_targets)
        btn_core.click(fn=lambda q: execute_suite_stream("core", q), inputs=[chk_quick], outputs=out_targets)
        btn_reasoning.click(fn=lambda q: execute_suite_stream("reasoning", q), inputs=[chk_quick], outputs=out_targets)
        btn_multilingual.click(fn=lambda q: execute_suite_stream("multilingual", q), inputs=[chk_quick], outputs=out_targets)
        btn_tools.click(fn=lambda q: execute_suite_stream("tools", q), inputs=[chk_quick], outputs=out_targets)
        btn_long.click(fn=lambda q: execute_suite_stream("long_context", q), inputs=[chk_quick], outputs=out_targets)
        btn_external.click(fn=lambda q: execute_suite_stream("external", q), inputs=[chk_quick], outputs=out_targets)
        btn_performance.click(fn=lambda q: execute_suite_stream("performance", q), inputs=[chk_quick], outputs=out_targets)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
