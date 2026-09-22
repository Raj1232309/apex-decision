"""
ApexDecision V8 Tool Result Verifier & Injection Firewall.
Validates tool outputs against schemas, detects stale or conflicting
returns across multiple tools, and enforces strict Control Plane vs.
Data Plane isolation to prevent prompt injection attacks.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResultVerificationVerdict:
    is_valid: bool
    sanitized_output: Dict[str, Any]
    conflict_detected: bool
    injection_detected: bool
    issues: List[str]


class ApexToolResultVerifier:
    """
    Verifies tool execution outputs and neutralizes malicious prompt injections.
    """

    INJECTION_TRIGGERS = [
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
        r"system\s*(?:alert|:)",
        r"override\s+(?:policy|safety)",
        r"grant\s+(?:all\s+)?permissions?",
        r"admin\s+access\s+granted",
        r"bypass\s+permission",
        r"disregard\s+(?:schema|instructions)",
        r"dump\s+secret",
        r"execute\s+tool\s+immediately",
        r"exfiltrate",
        r"do\s+not\s+verify"
    ]

    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_TRIGGERS]

    def sanitize_untrusted_content(self, data: Any) -> Tuple[Any, bool]:
        """
        Recursively scans strings inside tool outputs for prompt injection payloads
        and neutralizes them into passive data attributes.
        """
        injection_found = False
        if isinstance(data, str):
            clean = data
            for rgx in self.injection_regexes:
                if rgx.search(clean):
                    injection_found = True
                    clean = rgx.sub("[UNTRUSTED_INJECTION_DISARMED]", clean)
            return clean, injection_found
        elif isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                s_val, found = self.sanitize_untrusted_content(v)
                new_dict[k] = s_val
                if found:
                    injection_found = True
            return new_dict, injection_found
        elif isinstance(data, list):
            new_list = []
            for item in data:
                s_item, found = self.sanitize_untrusted_content(item)
                new_list.append(s_item)
                if found:
                    injection_found = True
            return new_list, injection_found
        return data, False

    def verify_result(
        self,
        tool_name: str,
        output_data: Optional[Dict[str, Any]],
        expected_schema: Optional[Dict[str, Any]] = None,
        prior_tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> ResultVerificationVerdict:
        """
        Validates structure and detects cross-tool contradictions.
        """
        issues = []
        if output_data is None:
            return ResultVerificationVerdict(
                is_valid=False,
                sanitized_output={},
                conflict_detected=False,
                injection_detected=False,
                issues=["Tool returned null payload."]
            )

        # 1. Sanitize prompt injections
        sanitized_data, injection_flag = self.sanitize_untrusted_content(output_data)
        if injection_flag:
            issues.append(f"Prompt injection attempt neutralized in output from tool '{tool_name}'.")

        # 2. Schema check
        if expected_schema and "required" in expected_schema:
            for req in expected_schema["required"]:
                if req not in sanitized_data:
                    issues.append(f"Output schema violation: missing required property '{req}'.")

        # 3. Conflict detection against prior tool returns
        conflict_flag = False
        if prior_tool_results:
            for prior in prior_tool_results:
                prior_name = prior.get("tool_name")
                prior_data = prior.get("data", {})
                # Check for direct key contradiction
                for k, v in sanitized_data.items():
                    if k in prior_data and prior_data[k] != v:
                        conflict_flag = True
                        issues.append(
                            f"Tool conflict detected between '{tool_name}' and '{prior_name}' on field '{k}': {v} vs {prior_data[k]}"
                        )

        is_valid = len(issues) == 0 or (injection_flag and not conflict_flag)
        return ResultVerificationVerdict(
            is_valid=is_valid,
            sanitized_output=sanitized_data,
            conflict_detected=conflict_flag,
            injection_detected=injection_flag,
            issues=issues
        )
