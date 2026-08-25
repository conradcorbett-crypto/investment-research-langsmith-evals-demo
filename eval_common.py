"""Shared helpers for the local evaluate scripts."""

from __future__ import annotations

import os
from typing import Any


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value == "*****":
        raise SystemExit(
            f"{name} is missing or still a placeholder. "
            "Copy example.env to .env and set real keys."
        )
    return value


def question_from_inputs(inputs: dict[str, Any]) -> str | None:
    question = inputs.get("question")
    if isinstance(question, str) and question.strip():
        return question
    for msg in inputs.get("messages") or []:
        if isinstance(msg, dict) and msg.get("type") == "human":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def build_replay_index(
    rows: list[dict[str, Any]], *, dataset_file: str
) -> dict[str, dict[str, Any]]:
    """Map each example question to its canned assistant output."""
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        question = (row.get("inputs") or {}).get("question")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"{dataset_file} row missing inputs.question")
        if question in index:
            raise SystemExit(f"Duplicate inputs.question in dataset: {question!r}")
        index[question] = row["actual_outputs"]
    return index


def print_experiment_result(results: Any, what_to_inspect: str) -> None:
    print(results)
    url = getattr(results, "experiment_url", None) or getattr(results, "url", None)
    if isinstance(url, str) and url.strip():
        print(f"Open in LangSmith: {url}")
    print(f"Done. In LangSmith, open the new experiment and inspect {what_to_inspect}.")
