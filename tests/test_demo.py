"""Tests for dataset schema, upload shaping, and policy evaluators."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators.research_policy import (  # noqa: E402
    citation_coverage_evaluator,
    no_personalized_advice_evaluator,
    perform_eval,
)
from upload_experiment import EXERCISES, load_rows, make_results  # noqa: E402


class DatasetSchemaTests(unittest.TestCase):
    def test_both_datasets_exist_with_twenty_rows(self):
        for exercise in EXERCISES:
            path = Path(exercise["jsonl_file"])
            self.assertTrue(path.is_file(), f"missing {path}")
            rows = load_rows(path)
            self.assertEqual(len(rows), 20, f"{path} should have 20 examples")

    def test_row_schema_and_categories(self):
        expected_categories = {
            "research-quality": {
                "well_grounded",
                "missing_citations",
                "unsupported_claims",
                "stale_info_presented_as_current",
                "uncertainty_handled_well",
                "personalized_recommendation",
            },
            "escalation-accuracy": {
                "correct_escalate",
                "correct_no_escalate",
                "incorrect_escalate",
                "incorrect_no_escalate",
                "incorrect_no_escalate_false_claim",
            },
        }
        for exercise in EXERCISES:
            rows = load_rows(Path(exercise["jsonl_file"]))
            cats = set()
            for row in rows:
                self.assertIn("inputs", row)
                self.assertIn("actual_outputs", row)
                self.assertIn("messages", row["inputs"])
                self.assertIn("messages", row["actual_outputs"])
                self.assertIsInstance(row["inputs"]["messages"], list)
                self.assertGreaterEqual(len(row["inputs"]["messages"]), 1)
                meta = row.get("metadata") or {}
                self.assertIn("category", meta)
                cats.add(meta["category"])
            self.assertTrue(
                expected_categories[exercise["dataset_name"]].issubset(cats),
                f"{exercise['dataset_name']} missing categories: "
                f"{expected_categories[exercise['dataset_name']] - cats}",
            )

    def test_every_row_has_reference_outputs(self):
        required_keys = {
            "research-quality": {
                "answer",
                "expected_sources",
                "expected_score",
                "label_rationale",
            },
            "escalation-accuracy": {
                "answer",
                "expected_action",
                "escalation_reason",
                "expected_score",
                "label_rationale",
            },
        }
        for exercise in EXERCISES:
            name = exercise["dataset_name"]
            for row in load_rows(Path(exercise["jsonl_file"])):
                expected = row.get("expected_outputs")
                self.assertIsInstance(expected, dict, name)
                self.assertTrue(
                    required_keys[name].issubset(expected),
                    f"{name} row missing reference keys: "
                    f"{required_keys[name] - set(expected)}",
                )
                self.assertTrue(expected["answer"].strip())
                self.assertIn(expected["expected_score"], (0, 1))

    def test_human_label_matches_category(self):
        passing_categories = {
            "well_grounded",
            "uncertainty_handled_well",
            "correct_escalate",
            "correct_no_escalate",
        }
        for exercise in EXERCISES:
            for row in load_rows(Path(exercise["jsonl_file"])):
                category = row["metadata"]["category"]
                self.assertEqual(
                    row["expected_outputs"]["expected_score"],
                    1 if category in passing_categories else 0,
                    f"label/category mismatch for {row['metadata']['scenario']}",
                )

    def test_flattened_fields_match_messages(self):
        for exercise in EXERCISES:
            for row in load_rows(Path(exercise["jsonl_file"])):
                first_human = next(
                    m["content"]
                    for m in row["inputs"]["messages"]
                    if m.get("type") == "human"
                )
                last_ai = next(
                    m["content"]
                    for m in reversed(row["actual_outputs"]["messages"])
                    if m.get("type") == "ai" and m.get("content")
                )
                self.assertEqual(row["inputs"]["question"], first_human)
                self.assertEqual(row["actual_outputs"]["answer"], last_ai)

    def test_escalation_actions_are_known_values(self):
        path = ROOT / "dataset" / "escalation_accuracy.jsonl"
        for row in load_rows(path):
            expected = row["expected_outputs"]
            self.assertIn(
                expected["expected_action"],
                {"escalate", "answer_from_public_sources"},
            )
            if expected["expected_action"] == "escalate":
                self.assertIn(
                    expected["escalation_reason"],
                    {"possible_mnpi", "personalized_advice", "trade_instruction"},
                )
            else:
                self.assertIsNone(expected["escalation_reason"])

    def test_jsonl_is_valid_utf8_json(self):
        for exercise in EXERCISES:
            path = Path(exercise["jsonl_file"])
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.assertIsInstance(obj, dict, f"{path}:{i}")


class UploadShapingTests(unittest.TestCase):
    def test_make_results_strips_metadata_and_adds_timing(self):
        path = Path(EXERCISES[0]["jsonl_file"])
        rows = load_rows(path)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        results = make_results(rows, base)
        self.assertEqual(len(results), len(rows))
        for row, result in zip(rows, results):
            self.assertIn("inputs", result)
            self.assertIn("actual_outputs", result)
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertNotIn("metadata", result)
            self.assertEqual(result["expected_outputs"], row["expected_outputs"])
            self.assertEqual(
                result["run_metadata"]["agent"], "investment-research-assistant"
            )
            self.assertEqual(
                result["run_metadata"]["category"], row["metadata"]["category"]
            )

    def test_load_rows_rejects_row_without_reference_outputs(self):
        bad = Path(self.enterContext(tempfile.TemporaryDirectory())) / "bad.jsonl"
        bad.write_text(
            json.dumps({"inputs": {"question": "hi"}, "actual_outputs": {"answer": "yo"}})
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_rows(bad)


class EvaluatorTests(unittest.TestCase):
    def test_citation_pass_and_fail(self):
        good = {
            "outputs": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "Revenue grew 8%. [source: NBRI 10-Q FY26 Q2]",
                    }
                ]
            }
        }
        bad = {
            "outputs": {
                "messages": [
                    {"type": "ai", "content": "Revenue grew 8% last quarter."}
                ]
            }
        }
        self.assertEqual(citation_coverage_evaluator(good)["score"], 1)
        self.assertEqual(citation_coverage_evaluator(bad)["score"], 0)

    def test_personalized_advice_detected(self):
        bad = {
            "outputs": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "You should buy Apex and allocate 15% of your portfolio.",
                    }
                ]
            }
        }
        good = {
            "outputs": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "Per the APXS 10-K, gross margin was 48%.",
                    }
                ]
            }
        }
        self.assertEqual(no_personalized_advice_evaluator(bad)["score"], 0)
        self.assertEqual(no_personalized_advice_evaluator(good)["score"], 1)

    def test_perform_eval_ui_shape(self):
        run = {
            "outputs": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "According to the filing, capacity was 4.2 GW.",
                    }
                ]
            }
        }
        result = perform_eval(run)
        self.assertTrue(result["has_citation"])
        self.assertTrue(result["no_personalized_advice"])

    def test_dataset_personalized_rows_fail_advice_check(self):
        path = ROOT / "dataset" / "research_quality.jsonl"
        rows = load_rows(path)
        flagged = [
            r
            for r in rows
            if r["metadata"]["category"] == "personalized_recommendation"
        ]
        self.assertGreaterEqual(len(flagged), 1)
        for r in flagged:
            score = no_personalized_advice_evaluator(
                {"outputs": r["actual_outputs"]}
            )["score"]
            self.assertEqual(
                score,
                0,
                f"expected advice violation for scenario={r['metadata']['scenario']}",
            )

    def test_well_grounded_rows_have_citations(self):
        path = ROOT / "dataset" / "research_quality.jsonl"
        rows = load_rows(path)
        grounded = [r for r in rows if r["metadata"]["category"] == "well_grounded"]
        self.assertGreaterEqual(len(grounded), 1)
        for r in grounded:
            score = citation_coverage_evaluator({"outputs": r["actual_outputs"]})[
                "score"
            ]
            self.assertEqual(
                score,
                1,
                f"expected citation for scenario={r['metadata']['scenario']}",
            )


class NoCustomerBrandingTests(unittest.TestCase):
    def test_repo_text_avoids_forbidden_brand_token(self):
        # Keep demos fully anonymized. Token assembled so this file is not a false positive.
        forbidden = "cap" + "ital" + " " + "gro" + "up"
        text_files = list(ROOT.rglob("*"))
        offenders = []
        for path in text_files:
            if not path.is_file():
                continue
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            if path.suffix.lower() not in {
                ".md",
                ".py",
                ".toml",
                ".jsonl",
                ".env",
                ".txt",
            } and path.name not in {".gitignore"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
            if forbidden in content:
                offenders.append(str(path))
        self.assertEqual(offenders, [], f"Forbidden brand token found in: {offenders}")


if __name__ == "__main__":
    unittest.main()
