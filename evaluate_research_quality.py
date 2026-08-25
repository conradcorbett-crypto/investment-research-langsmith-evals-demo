"""Replay canned research-quality runs and score per-example assertions.

Requires LANGSMITH_API_KEY and OPENAI_API_KEY.

Usage:
    uv run evaluate_research_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import evaluate

from eval_common import (
    build_replay_index,
    print_experiment_result,
    question_from_inputs,
    require_env,
)
from evaluators.research_policy import assertion_evaluator
from upload_experiment import load_rows

load_dotenv()

DATASET_NAME = "research-quality"
DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "research_quality.jsonl"


def main() -> None:
    require_env("LANGSMITH_API_KEY")
    require_env("OPENAI_API_KEY")

    rows = load_rows(DATASET_PATH)
    replay = build_replay_index(rows, dataset_file="research_quality.jsonl")

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = question_from_inputs(inputs)
        if question is None or question not in replay:
            raise KeyError(f"No synthetic output for question={question!r}")
        return replay[question]

    print(f"Evaluating {len(rows)} examples on dataset '{DATASET_NAME}'...")
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[assertion_evaluator],
        experiment_prefix="research-quality-assertions",
        metadata={"agent": "investment-research-assistant", "exercise": "2"},
    )
    print_experiment_result(results, "assertions_pass_rate")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
