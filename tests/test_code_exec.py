import sys

from routerbench.config import CodeExecConfig
from routerbench.models import DatasetItem
from routerbench.scoring.code_exec import CodeExecutionScorer

CONFIG = CodeExecConfig(timeout_s=5.0, memory_limit_mb=256, python_executable=sys.executable)

PALINDROME_ITEM = DatasetItem(
    id="code-1",
    prompt="write is_palindrome",
    metadata={
        "test_cases": [
            {"call": "is_palindrome('racecar')", "expected": True},
            {"call": "is_palindrome('hello')", "expected": False},
        ]
    },
)


async def test_code_exec_all_pass():
    scorer = CodeExecutionScorer(CONFIG)
    code = "def is_palindrome(s):\n    return s == s[::-1]\n"
    score = await scorer.score(PALINDROME_ITEM, code)
    assert score == 1.0


async def test_code_exec_extracts_markdown_fence():
    scorer = CodeExecutionScorer(CONFIG)
    code = "Here you go:\n```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"
    score = await scorer.score(PALINDROME_ITEM, code)
    assert score == 1.0


async def test_code_exec_partial_pass():
    scorer = CodeExecutionScorer(CONFIG)
    code = "def is_palindrome(s):\n    return True\n"  # always True -> fails the 'hello' case
    score = await scorer.score(PALINDROME_ITEM, code)
    assert score == 0.5


async def test_code_exec_syntax_error_scores_zero():
    scorer = CodeExecutionScorer(CONFIG)
    code = "def is_palindrome(s)\n    return s\n"  # missing colon
    score = await scorer.score(PALINDROME_ITEM, code)
    assert score == 0.0


async def test_code_exec_infinite_loop_times_out_and_scores_zero():
    scorer = CodeExecutionScorer(CodeExecConfig(timeout_s=1.0, memory_limit_mb=256, python_executable=sys.executable))
    code = "def is_palindrome(s):\n    while True:\n        pass\n"
    score = await scorer.score(PALINDROME_ITEM, code)
    assert score == 0.0


async def test_code_exec_no_test_cases_is_unscored():
    scorer = CodeExecutionScorer(CONFIG)
    item = DatasetItem(id="x", prompt="p")
    assert await scorer.score(item, "def f(): pass") is None
