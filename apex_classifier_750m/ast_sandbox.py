"""
Sandboxed AST Symbolic Compute Engine for Apex-748M.
Strictly whitelisted AST visitor with decimal precision, magnitude bounding, and denial-of-service protection.
Matches docs/APEX_750M_SPECIFICATION.md Section 6.
"""

import ast
import math
import decimal
from decimal import Decimal, Context, Overflow, DivisionByZero, InvalidOperation
from typing import Union

# Strict financial precision context with magnitude bounding
MATH_CONTEXT = Context(prec=28, Emax=999, Emin=-999, traps=[Overflow, DivisionByZero, InvalidOperation])

SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd
}


class SecurityError(Exception):
    """Raised when an expression attempts unsafe operations."""
    pass


class SandboxedMathEvaluator(ast.NodeVisitor):
    def __init__(self, max_steps: int = 100):
        self.steps = 0
        self.max_steps = max_steps

    def evaluate(self, expr_str: str) -> Decimal:
        """Parses and evaluates mathematical expression within a single context activation."""
        self.steps = 0
        try:
            parsed = ast.parse(expr_str.strip(), mode='eval')
        except SyntaxError as e:
            raise ValueError(f"SYNTAX_ERROR: {str(e)}")

        with decimal.localcontext(MATH_CONTEXT):
            return self.visit(parsed)

    def visit(self, node):
        self.steps += 1
        if self.steps > self.max_steps:
            raise ValueError("MATH_EVAL_STEP_LIMIT_EXCEEDED")
        if type(node) not in SAFE_NODES:
            raise SecurityError(f"DISALLOWED_NODE: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        # 1. Reject bool (bool is a subclass of int in Python)
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("DISALLOWED_CONSTANT_TYPE")
        # 2. Reject oversized literals first (prevents float overflow in math.isfinite for huge ints)
        if abs(node.value) > 1e15:
            raise OverflowError("CONSTANT_MAGNITUDE_EXCEEDED")
        # 3. Reject NaN and Infinities (safe to call since abs(value) <= 1e15)
        if not math.isfinite(node.value):
            raise ValueError("NON_FINITE_NUMERIC_CONSTANT")
        return Decimal(str(node.value))

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise SecurityError(f"DISALLOWED_UNARY_OP: {type(node.op).__name__}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        try:
            if op_type is ast.Add:
                result = left + right
            elif op_type is ast.Sub:
                result = left - right
            elif op_type is ast.Mult:
                result = left * right
            elif op_type is ast.Div:
                if right == 0:
                    raise ZeroDivisionError("MATH_ZERO_DIV")
                result = left / right
            elif op_type is ast.FloorDiv:
                if right == 0:
                    raise ZeroDivisionError("MATH_ZERO_DIV")
                result = (left // right)
            elif op_type is ast.Mod:
                if right == 0:
                    raise ZeroDivisionError("MATH_ZERO_DIV")
                result = left % right
            else:
                raise SecurityError(f"OPERATOR_NOT_PERMITTED: {op_type.__name__}")

            if abs(result) > Decimal('1e30'):
                raise OverflowError("INTERMEDIATE_MAGNITUDE_EXCEEDED")
            return result
        except (Overflow, DivisionByZero, InvalidOperation) as e:
            raise ArithmeticError(f"ARITHMETIC_FAULT: {str(e)}")
