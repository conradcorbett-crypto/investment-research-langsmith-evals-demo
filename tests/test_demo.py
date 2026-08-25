"""Tests for dataset schema, upload shaping, and assertion evaluators."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators.research_policy import (  # noqa: E402
    assertion_evaluator,
    has_citation,
    has_prohibited_recommendation,
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
                "assertions",
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

    def test_research_quality_assertions_shape(self):
        path = ROOT / "dataset" / "research_quality.jsonl"
        for row in load_rows(path):
            assertions = row["expected_outputs"]["assertions"]
            self.assertIsInstance(assertions, list)
            self.assertGreaterEqual(len(assertions), 1)
            for assertion in assertions:
                self.assertIn("key", assertion)
                self.assertIn("comment", assertion)
                self.assertTrue(str(assertion["key"]).strip())
                self.assertTrue(str(assertion["comment"]).strip())

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
        results = make_results(rows, base, EXERCISES[0]["dataset_name"])
        self.assertEqual(len(results), len(rows))
        for row, result in zip(rows, results):
            self.assertIn("inputs", result)
            self.assertIn("actual_outputs", result)
            self.assertIn("start_time", result)
            self.assertIn("end_time", result)
            self.assertNotIn("metadata", result)
            self.assertEqual(result["expected_outputs"], row["expected_outputs"])
            self.assertIn("assertions", result["expected_outputs"])
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


class AssertionEvaluatorTests(unittest.TestCase):
    def test_all_yes_scores_one(self):
        run = {"outputs": {"answer": "Revenue +8%. [source: NBRI 10-Q]"}}
        example = {
            "inputs": {"question": "Summarize revenue."},
            "outputs": {
                "assertions": [
                    {"key": "must_cite_source", "comment": "Must cite."},
                    {"key": "must_only_use_retrieved_evidence", "comment": "Grounded."},
                ]
            },
        }
        with patch(
            "evaluators.research_policy._judge", return_value=1.0
        ) as mock_judge:
            result = assertion_evaluator(run, example)
        self.assertEqual(result["key"], "assertions_pass_rate")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(mock_judge.call_count, 2)
        self.assertIn("2/2", result["comment"])

    def test_mixed_scores_fraction(self):
        run = {"outputs": {"answer": "You should buy NBRI."}}
        example = {
            "inputs": {"question": "Should I buy?"},
            "outputs": {
                "assertions": [
                    {"key": "a", "comment": "one"},
                    {"key": "b", "comment": "two"},
                    {"key": "c", "comment": "three"},
                ]
            },
        }
        with patch(
            "evaluators.research_policy._judge", side_effect=[1.0, 0.0, 1.0]
        ):
            result = assertion_evaluator(run, example)
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertIn("2/3", result["comment"])

    def test_empty_assertions_score_one(self):
        run = {"outputs": {"answer": "anything"}}
        example = {"inputs": {"question": "q"}, "outputs": {"assertions": []}}
        result = assertion_evaluator(run, example)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("no assertions", result["comment"])

    def test_missing_answer_scores_zero(self):
        run = {"outputs": {"answer": ""}}
        example = {
            "inputs": {"question": "q"},
            "outputs": {
                "assertions": [{"key": "must_cite_source", "comment": "cite"}]
            },
        }
        result = assertion_evaluator(run, example)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("no answer", result["comment"])

    def test_reads_expected_outputs_alias(self):
        run = {"outputs": {"answer": "ok [source: filing]"}}
        example = {
            "inputs": {"question": "q"},
            "expected_outputs": {
                "assertions": [{"key": "must_cite_source", "comment": "cite"}]
            },
        }
        with patch("evaluators.research_policy._judge", return_value=1.0):
            result = assertion_evaluator(run, example)
        self.assertEqual(result["score"], 1.0)

    def test_helpers_still_detect_citation_and_advice(self):
        self.assertTrue(has_citation("Per the 10-K, margin was 48%."))
        self.assertFalse(has_citation("Margin was 48%."))
        self.assertTrue(
            has_prohibited_recommendation("You should buy Apex and allocate 15%.")
        )
        self.assertFalse(
            has_prohibited_recommendation("Per the APXS 10-K, gross margin was 48%.")
        )


class EvalCommonTests(unittest.TestCase):
    def test_question_from_inputs_prefers_question_field(self):
        from eval_common import question_from_inputs

        self.assertEqual(
            question_from_inputs({"question": "What is revenue?"}),
            "What is revenue?",
        )

    def test_build_replay_index_rejects_duplicates(self):
        from eval_common import build_replay_index

        rows = [
            {
                "inputs": {"question": "q"},
                "actual_outputs": {"answer": "a"},
            },
            {
                "inputs": {"question": "q"},
                "actual_outputs": {"answer": "b"},
            },
        ]
        with self.assertRaises(SystemExit):
            build_replay_index(rows, dataset_file="dup.jsonl")

    def test_escalation_eval_script_wires_the_judge(self):
        src = (ROOT / "evaluate_escalation_accuracy.py").read_text(encoding="utf-8")
        self.assertIn("evaluators=[escalation_accuracy_evaluator]", src)


class EscalationEvaluatorTests(unittest.TestCase):
    def test_parses_json_score(self):
        from evaluators.escalation_accuracy import escalation_accuracy_evaluator

        run = {
            "outputs": {
                "answer": "Escalated to compliance.",
                "messages": [
                    {"type": "human", "content": "Buy this for me"},
                    {
                        "type": "ai",
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "escalate_to_compliance",
                                "args": {"reason": "trade_instruction"},
                            }
                        ],
                    },
                    {"type": "ai", "content": "Escalated to compliance."},
                ],
            }
        }
        example = {"inputs": {"question": "Buy this for me"}, "outputs": {}}

        class _Msg:
            content = '{"score": 1, "reasoning": "correct escalate"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        with patch("evaluators.escalation_accuracy._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _Resp()
            result = escalation_accuracy_evaluator(run, example)
        self.assertEqual(result["key"], "escalation_accuracy")
        self.assertEqual(result["score"], 1)
        self.assertIn("correct escalate", result["comment"])

    def test_missing_outputs_score_zero(self):
        from evaluators.escalation_accuracy import escalation_accuracy_evaluator

        result = escalation_accuracy_evaluator({"outputs": {}}, {"inputs": {}})
        self.assertEqual(result["score"], 0)


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
