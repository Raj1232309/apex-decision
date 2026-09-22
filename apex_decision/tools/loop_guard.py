"""
ApexDecision V8 Loop Guard & Cycle Detector.
Prevents infinite tool calling loops (e.g., tool A -> tool B -> tool A),
enforces invocation budgets, and detects redundant repeated calls.
"""

from typing import Dict, Any, List, Set, Tuple
import hashlib
import json


class ApexLoopGuard:
    """
    Deterministic loop and cycle prevention guard for tool orchestration.
    """

    def __init__(self, max_total_calls: int = 5, max_repeated_calls: int = 2):
        self.max_total_calls = max_total_calls
        self.max_repeated_calls = max_repeated_calls
        self.call_history: List[Dict[str, Any]] = []
        self.call_signatures: Set[str] = set()

    def reset(self) -> None:
        self.call_history.clear()
        self.call_signatures.clear()

    @staticmethod
    def compute_call_signature(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Deterministic hash signature for tool call."""
        raw = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def record_and_validate(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates if the proposed tool call is permissible under loop safety rules.
        """
        # 1. Total budget check
        if len(self.call_history) >= self.max_total_calls:
            return False, f"Total tool call limit ({self.max_total_calls}) exceeded."

        sig = self.compute_call_signature(tool_name, arguments)

        # 2. Repeated exact call check
        repeat_count = sum(1 for c in self.call_history if c["signature"] == sig)
        if repeat_count >= self.max_repeated_calls:
            return False, f"Duplicate tool call loop detected: '{tool_name}' invoked {repeat_count + 1} times with identical arguments."

        # 3. Ping-pong cycle detection (A -> B -> A -> B)
        history_names = [c["name"] for c in self.call_history]
        if len(history_names) >= 3:
            # Check for alternating 2-cycle
            if history_names[-2] == tool_name and history_names[-1] != tool_name and history_names[-3] == tool_name:
                return False, f"Alternating cycle pattern detected between tools: {' -> '.join(history_names[-3:] + [tool_name])}."

        self.call_history.append({
            "name": tool_name,
            "args": arguments,
            "signature": sig
        })
        self.call_signatures.add(sig)
        return True, ""
