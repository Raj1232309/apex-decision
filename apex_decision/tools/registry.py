"""
ApexDecision V8 Native Tool Registry.
Manages tool declarations, JSON schemas, permission attributes,
risk classifications, and scales up to 1,000 tools without retraining.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    permission_mode: str = "read_only"  # "read_only", "filesystem", "network", "database", "messaging", "action"
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    requires_confirmation: bool = False
    idempotent: bool = True
    timeout_seconds: float = 5.0
    max_retries: int = 2
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": {
                "mode": self.permission_mode,
                "risk": self.risk_level,
                "requires_confirmation": self.requires_confirmation
            },
            "idempotent": self.idempotent,
            "timeout_seconds": self.timeout_seconds
        }
        return d


class ApexToolRegistry:
    """
    Central catalog of registered tools in ApexDecision V8.
    Supports scaling from 10 to 1,000 tools seamlessly.
    """

    def __init__(self, register_defaults: bool = True):
        self._tools: Dict[str, ToolDefinition] = {}
        if register_defaults:
            self.register_standard_tools()

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_catalog(self) -> Dict[str, str]:
        """Returns option descriptions for Choice-based selection."""
        return {
            name: f"{tool.description} (Permissions: {tool.permission_mode}, Risk: {tool.risk_level})"
            for name, tool in self._tools.items()
        }

    def count(self) -> int:
        return len(self._tools)

    def register_standard_tools(self) -> None:
        """Registers a standard suite of 10 system tools with valid schemas and handlers."""
        # 1. Calculator
        self.register(ToolDefinition(
            name="calculator",
            description="Calculates exact mathematical expressions, multi-hop equations, and arithmetic operations.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to compute"}
                },
                "required": ["expression"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "number"},
                    "expression": {"type": "string"}
                },
                "required": ["result"]
            },
            permission_mode="read_only",
            risk_level="low",
            handler=lambda args: {
                "result": float(eval(args["expression"].replace("^", "**"), {"__builtins__": None}, {})),
                "expression": args["expression"]
            }
        ))

        # 2. Weather Lookup
        self.register(ToolDefinition(
            name="weather_lookup",
            description="Retrieves current weather, temperature, and meteorological conditions for a city or region.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or territory name"}
                },
                "required": ["location"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "temperature_c": {"type": "number"},
                    "condition": {"type": "string"}
                },
                "required": ["location", "temperature_c", "condition"]
            },
            permission_mode="read_only",
            risk_level="low",
            handler=lambda args: {
                "location": args["location"],
                "temperature_c": 18.5,
                "condition": "Clear skies with light breeze"
            }
        ))

        # 3. Web Search
        self.register(ToolDefinition(
            name="web_search",
            description="Searches the web for up-to-date facts, current news, and external documentation.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "total_found": {"type": "integer"}
                },
                "required": ["results"]
            },
            permission_mode="network",
            risk_level="low",
            handler=lambda args: {
                "query": args["query"],
                "total_found": 3,
                "results": [
                    {"title": f"Fact Sheet for {args['query']}", "snippet": f"Verified factual data regarding {args['query']}."}
                ]
            }
        ))

        # 4. Database Query
        self.register(ToolDefinition(
            name="database_query",
            description="Queries internal databases for user records, system logs, or transaction histories.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL or filter query"}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "rows": {"type": "array"}
                },
                "required": ["rows"]
            },
            permission_mode="database",
            risk_level="medium",
            handler=lambda args: {
                "rows": [{"id": 101, "record_type": "standard_entry", "query_matched": args["query"]}]
            }
        ))

        # 5. File Reader
        self.register(ToolDefinition(
            name="file_reader",
            description="Reads local filesystem files and returns document text.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"}
                },
                "required": ["path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"}
                },
                "required": ["content"]
            },
            permission_mode="filesystem",
            risk_level="medium",
            handler=lambda args: {
                "path": args["path"],
                "content": f"[File content placeholder for path: {args['path']}]"
            }
        ))

        # 6. File Writer
        self.register(ToolDefinition(
            name="file_writer",
            description="Creates or overwrites files on the local filesystem.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            },
            permission_mode="filesystem",
            risk_level="high",
            requires_confirmation=True,
            handler=lambda args: {
                "status": "written",
                "bytes": len(args["content"])
            }
        ))

        # 7. Execute Transaction
        self.register(ToolDefinition(
            name="execute_transaction",
            description="Executes a financial balance transfer, wire transaction, or debit action.",
            input_schema={
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["account_id", "amount"]
            },
            permission_mode="action",
            risk_level="critical",
            requires_confirmation=True,
            handler=lambda args: {
                "transaction_id": "tx_998877",
                "status": "COMPLETED",
                "amount": args["amount"],
                "account_id": args["account_id"]
            }
        ))

        # 8. Send Notification
        self.register(ToolDefinition(
            name="send_notification",
            description="Dispatches external alerts, emails, or SMS notifications to operators.",
            input_schema={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["recipient", "message"]
            },
            permission_mode="messaging",
            risk_level="medium",
            handler=lambda args: {
                "status": "SENT",
                "recipient": args["recipient"]
            }
        ))

        # 9. System Status
        self.register(ToolDefinition(
            name="system_status",
            description="Retrieves current CPU load, memory utilization, disk space, and daemon status.",
            input_schema={
                "type": "object",
                "properties": {
                    "component": {"type": "string"}
                },
                "required": []
            },
            permission_mode="read_only",
            risk_level="low",
            handler=lambda args: {
                "status": "HEALTHY",
                "cpu_percent": 14.2,
                "memory_percent": 38.1
            }
        ))

        # 10. Get Current Time
        self.register(ToolDefinition(
            name="get_current_time",
            description="Retrieves the current date, UTC timestamp, and local system time.",
            input_schema={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": []
            },
            permission_mode="read_only",
            risk_level="low",
            handler=lambda args: {
                "utc_iso": "2026-09-22T14:20:00Z",
                "timezone": args.get("timezone", "UTC")
            }
        ))

