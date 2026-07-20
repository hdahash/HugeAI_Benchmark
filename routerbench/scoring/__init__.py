from __future__ import annotations

from routerbench.client import RouterClient
from routerbench.config import BenchmarkConfig
from routerbench.scoring.base import NullScorer, Scorer
from routerbench.scoring.code_exec import CodeExecutionScorer
from routerbench.scoring.composite import CompositeScorer
from routerbench.scoring.heuristic import HeuristicScorer
from routerbench.scoring.llm_judge import LLMJudgeScorer

__all__ = [
    "Scorer",
    "NullScorer",
    "HeuristicScorer",
    "CodeExecutionScorer",
    "LLMJudgeScorer",
    "CompositeScorer",
    "build_scorer",
]


def build_scorer(config: BenchmarkConfig, judge_client: RouterClient | None) -> Scorer:
    mode = config.scorer.mode
    if mode == "none":
        return NullScorer()
    if mode == "heuristic":
        return HeuristicScorer()
    if mode != "auto":
        raise ValueError(f"Unknown scorer mode: {mode}")

    heuristic = HeuristicScorer()
    code_exec = CodeExecutionScorer(config.code_exec) if config.code_exec.enabled else None
    judge = LLMJudgeScorer(judge_client, config.judge) if (config.judge.enabled and judge_client is not None) else None
    return CompositeScorer(heuristic=heuristic, code_exec=code_exec, judge=judge)
