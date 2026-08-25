"""Replay canned escalation runs and score them with the starter LLM judge.

Uses the intentionally incomplete rubric in
``judge_prompts/escalation_accuracy.md``. Requires LANGSMITH_API_KEY and
OPENAI_API_KEY.

Usage:
    uv run evaluate_escalation_accuracy.py
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
from evaluators.escalation_accuracy import escalation_accuracy_evaluator
from upload_experiment import load_rows

load_dotenv()

DATASET_NAME = "escalation-accuracy"
DATASET_PATH = (
    Path(__file__).resolve().parent / "dataset" / "escalation_accuracy.jsonl"
)


def main() -> None:
    require_env("LANGSMITH_API_KEY")
    require_env("OPENAI_API_KEY")

    rows = load_rows(DATASET_PATH)
    replay = build_replay_index(rows, dataset_file="escalation_accuracy.jsonl")

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = question_from_inputs(inputs)
        if question is None or question not in replay:
            raise KeyError(f"No synthetic output for question={question!r}")
        return replay[question]

    print(f"Evaluating {len(rows)} examples on dataset '{DATASET_NAME}'...")
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[escalation_accuracy_evaluator],
        experiment_prefix="escalation-accuracy-judge",
        metadata={"agent": "investment-research-assistant", "exercise": "3"},
    )
    print_experiment_result(
        results,
        "escalation_accuracy, then compare it to reference_outputs.expected_score",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
