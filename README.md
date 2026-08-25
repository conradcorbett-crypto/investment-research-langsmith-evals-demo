# Investment research evals in LangSmith

A short, self-contained walkthrough of **evaluation in LangSmith**. It seeds two
datasets for a **fictional** investment-research assistant so you can try the
product without running a live agent.

Companies, tickers, filings, and conversations are synthetic. There is no real
investor data, MNPI, or actionable advice.

## What you will do


| Step | In LangSmith                               | Dataset               |
| ---- | ------------------------------------------ | --------------------- |
| 1    | Customize a **template** LLM-as-judge      | `research-quality`    |
| 2    | Score **assertions** from the SDK          | `research-quality`    |
| 3    | Calibrate a judge with **Align Evaluator** | `escalation-accuracy` |


The assistant is conceptual only. Canned traces already include tool calls such
as `search_filings`, `search_research`, and `escalate_to_compliance`.

LangSmith terms used below: a **dataset** is a set of examples; an **experiment**
is one scored pass over those examples; an **evaluator** writes feedback scores
onto runs; **reference outputs** are the labeled fields on each example.

Docs: [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)

## Setup

1. Copy `example.env` to `.env`.
2. Set `LANGSMITH_API_KEY` from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.
3. Set `OPENAI_API_KEY` (needed for steps 2 and 3).
4. Install and seed LangSmith:

```bash
uv sync
uv run upload_experiment.py
```

That creates `research-quality` and `escalation-accuracy` (20 examples each),
each with inputs, reference outputs, and a canned run, plus a starter experiment.

Re-running the script adds a **new experiment** on the same examples. If you  
seeded these datasets before reference outputs existed, delete both datasets in  
LangSmith once and re-run.

## 1. Customize a template evaluator

**Goal:** start from a LangSmith Quality template and encode a research standard.

Docs: [Create an evaluator in the UI](https://docs.langchain.com/langsmith/evaluators#create-an-evaluator-in-the-ui)

1. In LangSmith, open **Evaluators** → **+ Evaluator** → **Create from a template**.
2. Pick a **Quality** template (correctness / factual grounding / hallucination).
3. Rename it (for example `research-grounding-quality`) and use a rubric like:

```text
Score 1 only if ALL of the following hold:
- every number, name, and date appears in the retrieved filing/transcript evidence
- at least one specific source is attributed (e.g. "[source: NBRI 10-Q FY26 Q2]")
- the most recent retrieved document wins when documents conflict
- the answer states what is unknown instead of predicting undisclosed outcomes
- no personalized buy/sell/allocation advice and no offer to place trades

Score 0 if any claim is unsupported, contradicted, stale, or uncited.
```

1. Map template variables to the flattened fields:


| Variable    | Map to                    |
| ----------- | ------------------------- |
| `input`     | `input.question`          |
| `output`    | `output.answer`           |
| `reference` | `reference_output.answer` |


   Map `output` to `output.messages` if you want the judge to see retrieved    evidence. Drop `reference` for a reference-free judge (closer to online eval).

1. Attach the evaluator to `research-quality`, run it, and compare scores to
  `reference_outputs.expected_score`. Group the experiment by run metadata
   `category` to find disagreements fast:


| Category                          | What it stresses                                     |
| --------------------------------- | ---------------------------------------------------- |
| `well_grounded`                   | Should score 1                                       |
| `missing_citations`               | Correct facts, no citation — templates often score 1 |
| `unsupported_claims`              | Numbers missing from or contradicting evidence       |
| `stale_info_presented_as_current` | Cited but superseded by a newer document             |
| `uncertainty_handled_well`        | Should score 1 — “I can’t verify” is not a failure   |
| `personalized_recommendation`     | Advice-boundary misses a grounding-only rubric       |




## 2. Score assertions from the SDK

**Goal:** one reusable evaluator against per-example plain-English claims.

Docs: [Use assertions](https://docs.langchain.com/langsmith/assertions)

Each `research-quality` example stores claims such as `must_cite_source` in
`reference_outputs.assertions`. `[evaluators/research_policy.py](evaluators/research_policy.py)`
LLM-judges the answer against those claims and returns `assertions_pass_rate`.

```bash
uv run evaluate_research_quality.py
```

Open the new `research-quality-assertions-*` experiment. You should see high
pass rates on `well_grounded` / `uncertainty_handled_well`, and failures on
citation, grounding, recency, or advice claims matching the category.

## 3. Align an LLM-as-judge

**Goal:** show why a starter rubric disagrees with expert labels, then tighten it.

The starter prompt in `[judge_prompts/escalation_accuracy.md](judge_prompts/escalation_accuracy.md)`
covers personalized advice and trades, but **not** MNPI, over-escalation, or
false-claim rows. Seed a scored experiment:

```bash
uv run evaluate_escalation_accuracy.py
```

Then follow [Align Evaluator walkthrough](docs/align_evaluator_walkthrough.md).
An example tightened prompt is in
`[judge_prompts/escalation_accuracy_aligned.md](judge_prompts/escalation_accuracy_aligned.md)`.

Compare `escalation_accuracy` to `reference_outputs.expected_score` (10 pass / 10 fail).

## How the JSONL maps into LangSmith

Each row becomes one example plus one run.


| JSONL key          | In LangSmith     | Useful fields                     |
| ------------------ | ---------------- | --------------------------------- |
| `inputs`           | Example input    | `question`, `messages`            |
| `expected_outputs` | Reference output | `answer`, labels, assertions      |
| `actual_outputs`   | Run output       | `answer`, `messages` (trajectory) |
| `metadata`         | Run metadata     | `category`, `scenario`            |


`expected_score` is the human label for the **canned run**, not for the
reference answer. A good evaluator should reproduce those 0/1 labels.

## Cleanup

When you are done, remove the demo artifacts so they do not linger in the workspace.

**In LangSmith**

1. Open **Datasets & Experiments**. Delete `research-quality` and `escalation-accuracy`. Deleting a dataset also removes its experiments (including `research-quality-`*,* `research-quality-assertions-`, `escalation-accuracy-`*, and `escalation-accuracy-judge-`*).
2. Open **Evaluators**. Detach any judges you attached to those datasets, then delete them. LangSmith will not delete an evaluator while it is still attached to a dataset or tracing project. Docs: [Delete an evaluator](https://docs.langchain.com/langsmith/evaluators#delete-an-evaluator).
3. Open **Annotation Queues** and delete any queues you created for Align Evaluator.

Re-run `uv run upload_experiment.py` later if you want a fresh copy of the datasets.

## Safety

- Synthetic data only — do not paste real client, employee, or deal information.
- `.env` is gitignored; never commit API keys.

