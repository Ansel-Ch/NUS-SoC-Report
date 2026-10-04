"""Countdown task: data splits and answer scoring shared by SFT, OPD and evaluation."""

import ast
import operator
import re
from collections import Counter

from datasets import load_dataset

DATASET = "HuggingFaceTB/Countdown-Task-GOLD"
TEACHER_CONFIG = "verified_Qwen3-4B-Instruct-2507"
SFT_SIZE = 10_000  # first SFT_SIZE verified rows train SFT; the rest supply OPD prompts
SEED = 0

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def sft_split():
    """Verified teacher solutions used for the SFT warm-start (and the SFT baseline)."""
    ds = load_dataset(DATASET, TEACHER_CONFIG, split="train").shuffle(seed=SEED)
    return ds.select(range(SFT_SIZE))


def opd_split():
    """Prompts for on-policy distillation, disjoint from the SFT split.

    Rows keep the teacher's assistant message so the dataset is in the conversational
    language-modelling format GOLD expects; with lmbda=1 the completion is replaced by
    a student rollout and never trained on.
    """
    ds = load_dataset(DATASET, TEACHER_CONFIG, split="train").shuffle(seed=SEED)
    return ds.select(range(SFT_SIZE, len(ds)))


def test_split(n=None):
    ds = load_dataset(DATASET, "test", split="test")
    return ds.select(range(n)) if n else ds


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError("disallowed expression")


def extract_answer(text):
    """Equation inside the last <answer> block, with any trailing '= result' removed."""
    matches = _ANSWER.findall(text)
    if not matches:
        return None
    eq = matches[-1].strip()
    eq = eq.replace("×", "*").replace("÷", "/").replace("−", "-")
    return eq.split("=")[0].strip()


def score(text, nums, target):
    """Classify a completion. Returns (correct, reason)."""
    eq = extract_answer(text)
    if eq is None:
        return False, "no_answer_tag"
    used = [int(n) for n in re.findall(r"\d+", eq)]
    if Counter(used) != Counter(nums):
        return False, "wrong_numbers"
    try:
        value = _eval_node(ast.parse(eq, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False, "unparsable"
    if abs(value - target) > 1e-5:
        return False, "wrong_value"
    return True, "correct"
