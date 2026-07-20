"""Executes model-generated code against dataset-provided test cases and
scores pass-rate. This is ground truth for verifiable coding tasks --
no judge, no proxy metric, just "did it produce the right answer".

SECURITY: this scorer runs whatever code the model under test produced.
Isolation here (python -I, stripped env, CPU/memory/wall-clock limits via
`resource`, a throwaway temp directory) is a best-effort local sandbox, not
a security boundary. Only run this against ephemeral/disposable
infrastructure with no reachable secrets or sensitive network access --
never on shared hosts. For anything beyond a benchmark demo, run it inside
a container or microVM instead.

Dataset schema (in DatasetItem.metadata):
  "test_cases": [{"call": "is_palindrome('racecar')", "expected": true}, ...]
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from routerbench.config import CodeExecConfig
from routerbench.models import DatasetItem

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_RESULT_PREFIX = "ROUTERBENCH_RESULT:"

_HARNESS_TEMPLATE = """\
import json

{code}

def __routerbench_main():
    with open("test_cases.json") as f:
        test_cases = json.load(f)
    passed = 0
    for case in test_cases:
        try:
            actual = eval(case["call"])
            if actual == case["expected"]:
                passed += 1
        except Exception:
            pass
    print("{result_prefix}" + json.dumps({{"passed": passed, "total": len(test_cases)}}))

__routerbench_main()
"""


def _extract_code(content: str) -> str:
    match = _CODE_FENCE_RE.search(content)
    return match.group(1) if match else content


def _set_resource_limits(memory_limit_mb: int) -> None:
    try:
        import resource
    except ImportError:
        return  # not available on this platform (e.g. Windows)

    mem_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def _run_sandboxed(code: str, test_cases: list[dict], config: CodeExecConfig) -> float:
    with tempfile.TemporaryDirectory(prefix="routerbench-codeexec-") as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "solution.py").write_text(
            _HARNESS_TEMPLATE.format(code=code, result_prefix=_RESULT_PREFIX)
        )
        (tmp_path / "test_cases.json").write_text(json.dumps(test_cases))

        preexec_fn = (
            (lambda: _set_resource_limits(config.memory_limit_mb)) if sys.platform != "win32" else None
        )
        try:
            proc = subprocess.run(
                [config.python_executable, "-I", "solution.py"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
                env={"PATH": "/usr/bin:/bin"},
                preexec_fn=preexec_fn,
            )
        except (subprocess.TimeoutExpired, OSError):
            return 0.0

        for line in proc.stdout.splitlines():
            if line.startswith(_RESULT_PREFIX):
                try:
                    result = json.loads(line[len(_RESULT_PREFIX):])
                    total = result.get("total", 0)
                    return (result["passed"] / total) if total else 0.0
                except (json.JSONDecodeError, KeyError, ZeroDivisionError):
                    return 0.0
        return 0.0  # code didn't run to completion (syntax error, crash, etc.)


class CodeExecutionScorer:
    def __init__(self, config: CodeExecConfig):
        self.config = config

    async def score(self, item: DatasetItem, content: str) -> float | None:
        test_cases = item.metadata.get("test_cases")
        if not test_cases:
            return None

        code = _extract_code(content)
        return await asyncio.to_thread(_run_sandboxed, code, test_cases, self.config)
