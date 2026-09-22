from apex_decision.tools.executor import ApexToolExecutor
from apex_decision.tools.loop_guard import ApexLoopGuard


def test_schema_rejects_missing_required_argument():
    ok, _ = ApexToolExecutor.validate_schema({}, {"required": ["location"], "properties": {"location": {"type": "string"}}})
    assert not ok


def test_loop_guard_blocks_repeated_calls():
    guard = ApexLoopGuard(max_repeated_calls=1)
    assert guard.record_and_validate("safe", {"x": 1})[0]
    assert not guard.record_and_validate("safe", {"x": 1})[0]
