"""Upload synthetic investment-research experiments to LangSmith.

Requires LANGSMITH_API_KEY in the environment (or a local .env file).
Does not print secret values.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
ALLOWED_ENDPOINT_HOSTS = {
    "api.smith.langchain.com",
    "eu.api.smith.langchain.com",
}


# Stable namespace so repeated uploads reuse the same dataset examples instead of
# appending a fresh copy of every row.
ROW_ID_NAMESPACE = uuid.UUID("6f3f1b9a-6d4f-5c2e-8a5b-1f9f6d2c7e41")

EXERCISES = [
    {
        "dataset_name": "research-quality",
        "jsonl_file": DATASET_DIR / "research_quality.jsonl",
        "experiment_prefix": "research-quality",
    },
    {
        "dataset_name": "escalation-accuracy",
        "jsonl_file": DATASET_DIR / "escalation_accuracy.jsonl",
        "experiment_prefix": "escalation-accuracy",
    },
]


def _require_api_key() -> str:
    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not api_key or api_key == "*****":
        raise SystemExit(
            "LANGSMITH_API_KEY is missing or still a placeholder. "
            "Copy example.env to .env and set a real key from smith.langchain.com."
        )
    return api_key


def _endpoint() -> str:
    endpoint = os.environ.get(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    ).rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(f"Invalid LANGSMITH_ENDPOINT: {endpoint!r}")
    host = parsed.hostname or ""
    if host not in ALLOWED_ENDPOINT_HOSTS and not host.endswith(
        ".langchain.com"
    ):
        raise SystemExit(
            f"Refusing unexpected LANGSMITH_ENDPOINT host {host!r}. "
            f"Allowed hosts: {sorted(ALLOWED_ENDPOINT_HOSTS)} or *.langchain.com"
        )
    return endpoint


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            missing = [
                key
                for key in ("inputs", "expected_outputs", "actual_outputs")
                if key not in row
            ]
            if missing:
                raise ValueError(
                    f"Row {line_no} in {path} is missing required key(s): "
                    f"{', '.join(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def make_results(
    rows: list[dict[str, Any]],
    base_time: datetime,
    dataset_name: str = "",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    offset = 0.0
    for index, row in enumerate(rows):
        meta = row.get("metadata") or {}
        row_start = base_time + timedelta(seconds=offset)
        latency = float(meta.get("latency_seconds", 7.0))
        row_end = row_start + timedelta(seconds=latency)
        offset += latency + 1.0
        row_key = meta.get("scenario") or str(index)
        results.append(
            {
                "row_id": str(
                    uuid.uuid5(ROW_ID_NAMESPACE, f"{dataset_name}:{row_key}")
                ),
                "inputs": row["inputs"],
                # Reference outputs land on the dataset example, so reference-based
                # evaluator templates and Align Evals have something to compare against.
                "expected_outputs": row["expected_outputs"],
                "actual_outputs": row["actual_outputs"],
                "start_time": row_start.isoformat(),
                "end_time": row_end.isoformat(),
                "run_metadata": {
                    "ls_model_name": "gpt-4o",
                    "ls_provider": "openai",
                    "agent": "investment-research-assistant",
                    "category": meta.get("category"),
                    "scenario": meta.get("scenario"),
                },
            }
        )
    return results


def upload_experiment(
    results: list[dict[str, Any]],
    experiment_name: str,
    dataset_name: str,
    base_time: datetime,
    *,
    api_key: str,
    endpoint: str,
) -> dict[str, Any]:
    last_end = max(r["end_time"] for r in results)
    end = datetime.fromisoformat(last_end) + timedelta(seconds=1)
    body = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "experiment_start_time": base_time.isoformat(),
        "experiment_end_time": end.isoformat(),
        "results": results,
    }
    resp = requests.post(
        f"{endpoint}/api/v1/datasets/upload-experiment",
        json=body,
        headers={"x-api-key": api_key},
        timeout=120,
    )
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    data = resp.json()
    print(f"  Dataset: {data['dataset']['name']} ({data['dataset']['id']})")
    print(f"  Experiment: {data['experiment']['name']} ({data['experiment']['id']})")
    return data


def main() -> None:
    api_key = _require_api_key()
    endpoint = _endpoint()
    base_time = datetime.now(timezone.utc)
    ts = base_time.strftime("%Y%m%d-%H%M%S")

    for exercise in EXERCISES:
        path = Path(exercise["jsonl_file"])
        rows = load_rows(path)
        print(f"Loaded {len(rows)} examples from {path}")
        print(f"\nUploading {len(rows)} examples to {exercise['dataset_name']}...")
        results = make_results(rows, base_time, exercise["dataset_name"])
        experiment_name = f"{exercise['experiment_prefix']}-{ts}"
        upload_experiment(
            results,
            experiment_name,
            exercise["dataset_name"],
            base_time,
            api_key=api_key,
            endpoint=endpoint,
        )


if __name__ == "__main__":
    main()
