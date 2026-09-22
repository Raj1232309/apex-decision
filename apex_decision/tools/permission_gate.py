"""
ApexDecision V8 Security & Permission Gate.
Enforces deterministic safety policies across tool execution modes:
read-only, network, filesystem, database, messaging, and financial actions.
The model can NEVER override or rewrite permission policies.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass


@dataclass
class PermissionCheckResult:
    allowed: bool
    requires_confirmation: bool
    reason: str
    risk_level: str


class ApexPermissionGate:
    """
    Deterministic security barrier between neural decision layer and system execution layer.
    """

    def __init__(
        self,
        allowed_modes: Optional[Set[str]] = None,
        max_allowed_risk: str = "medium",
        allow_external_side_effects: bool = False
    ):
        # Default allowed modes
        self.allowed_modes = allowed_modes or {"read_only", "network", "filesystem", "database"}
        self.max_allowed_risk = max_allowed_risk
        self.allow_external_side_effects = allow_external_side_effects

        self.risk_hierarchy = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3
        }

    def check_permission(self, tool_def: Any, confirmed_by_user: bool = False) -> PermissionCheckResult:
        """
        Evaluates whether a tool execution is authorized under the active policy.
        """
        mode = getattr(tool_def, "permission_mode", "read_only")
        risk = getattr(tool_def, "risk_level", "low")
        req_confirm = getattr(tool_def, "requires_confirmation", False)

        # 1. Mode authorization
        if mode not in self.allowed_modes:
            if not confirmed_by_user:
                return PermissionCheckResult(
                    allowed=False,
                    requires_confirmation=True if mode in ("action", "financial", "messaging") else False,
                    reason=f"Permission mode '{mode}' is disallowed by current policy without confirmation.",
                    risk_level=risk
                )

        # 2. Risk threshold check
        tool_risk_score = self.risk_hierarchy.get(risk, 2)
        policy_risk_score = self.risk_hierarchy.get(self.max_allowed_risk, 1)

        if tool_risk_score > policy_risk_score:
            if not confirmed_by_user:
                return PermissionCheckResult(
                    allowed=False,
                    requires_confirmation=True,
                    reason=f"Tool risk level '{risk}' exceeds maximum policy '{self.max_allowed_risk}' and requires explicit confirmation.",
                    risk_level=risk
                )

        # 3. Explicit confirmation gate
        if req_confirm and not confirmed_by_user:
            return PermissionCheckResult(
                allowed=False,
                requires_confirmation=True,
                reason="Tool requires explicit human confirmation before execution.",
                risk_level=risk
            )

        # 4. Action / Financial side-effect gate
        if mode in ("action", "financial") and not self.allow_external_side_effects and not confirmed_by_user:
            return PermissionCheckResult(
                allowed=False,
                requires_confirmation=True,
                reason="External side-effecting action disallowed without explicit confirmation.",
                risk_level=risk
            )

        return PermissionCheckResult(
            allowed=True,
            requires_confirmation=False,
            reason="Execution authorized under policy.",
            risk_level=risk
        )
