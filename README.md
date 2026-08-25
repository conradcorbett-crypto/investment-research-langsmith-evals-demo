# Investment Research — LangSmith Evals Demo

Standalone demo (forked conceptually from the Align Evals workshop template) that shows how an **investment research assistant** team can:

1. **Customize prebuilt / template evaluators** in LangSmith
2. **Author deterministic custom code evaluators**
3. **Calibrate an LLM-as-judge** with the **Align Evaluator** workflow

All companies, tickers, filings, and conversations are **fictional**. There is no real investor data, MNPI, or actionable advice.

## Demo agent (conceptual)

An internal research assistant that can:


| Tool                     | Purpose                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `search_filings`         | Retrieve excerpts from public 10-K / 10-Q / 8-K / proxy filings    |
| `search_research`        | Retrieve transcripts / IR decks                                    |
| `escalate_to_compliance` | Hand off personalized advice, trade instructions, or possible MNPI |


Policy highlights for demos:

- Ground answers in cited public sources
- Do **not** give personalized buy/sell/allocate advice
- Do **not** place trades
- Escalate possible MNPI / confidential deal materials



## Repo layout

```
dataset/
  research_quality.jsonl      # grounding, citations, uncertainty, advice boundaries
  escalation_accuracy.jsonl   # escalate vs answer from public research
evaluators/
  research_policy.py          # deterministic citation + advice checks
judge_prompts/
  escalation_accuracy.md      # starter LLM judge (intentionally incomplete)
docs/
  align_evaluator_walkthrough.md
upload_experiment.py          # seed datasets + starter experiments in LangSmith
```

## Dataset schema

Each JSONL row becomes one dataset example plus one experiment run. Rows carry both a
message-array view (for trajectory-style and code evaluators) and a flattened
string view (so evaluator templates can map variables to a single field).

| Row key            | Lands in LangSmith as | Contents                                                    |
| ------------------ | --------------------- | ----------------------------------------------------------- |
| `inputs`           | Example input         | `question` (string) + `messages` (array)                     |
| `expected_outputs` | **Reference output**  | `answer` (string) + labels, see below                        |
| `actual_outputs`   | Run output            | `answer` (string) + full `messages` trajectory               |
| `metadata`         | Run metadata          | `category`, `scenario`, `latency_seconds`                     |

Reference output fields:

| Field                          | Dataset             | Purpose                                                          |
| ------------------------------ | ------------------- | ---------------------------------------------------------------- |
| `answer`                       | both                | The response a policy-compliant assistant should have given      |
| `expected_sources`             | research-quality    | Filings/transcripts the answer must be grounded in               |
| `expected_action`              | escalation-accuracy | `escalate` or `answer_from_public_sources`                        |
| `escalation_reason`            | escalation-accuracy | `possible_mnpi`, `personalized_advice`, `trade_instruction`, null |
| `expected_score`               | both                | Human ground-truth label: 1 = run passed policy, 0 = it failed   |
| `label_rationale`              | both                | One-line reason for the label, for judge alignment               |

`expected_score` is the human label for the **run that was recorded**, not for the
reference answer. It is what you grade your evaluator against: a good judge should
reproduce these 0/1 labels. `research-quality` is labeled 8 pass / 12 fail, and
`escalation-accuracy` is 10 / 10.

## Setup



### 1. Environment

```bash
cd investment-research-langsmith-evals-demo
cp example.env .env
```

Set `LANGSMITH_API_KEY` in `.env` from [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys.

Optional: `LANGSMITH_ENDPOINT` (defaults to `https://api.smith.langchain.com`).

### 2. Install tooling

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```



### 3. Upload seeded experiments

```bash
uv run upload_experiment.py
# or: python upload_experiment.py
```

Creates datasets:

- `research-quality`
- `escalation-accuracy`

…each with 20 examples that have **inputs, reference outputs, and a run**, plus a
timestamped starter experiment.

Rows upload with stable `row_id`s, so re-running the script adds a new experiment
against the same examples rather than duplicating them. If you seeded these datasets
before reference outputs existed, delete both datasets in LangSmith once and re-run —
otherwise you will see 20 examples with empty reference outputs alongside the new ones.

### 4. Run local tests (no LangSmith upload)

```bash
uv run python -m unittest discover -s tests -v
```

---



## Exercise 1 — Customize a prebuilt / template evaluator

**Goal:** Show how to start from LangSmith’s ready-made evaluator templates and tailor them to research grounding quality.

Docs: [Create an evaluator in the UI](https://docs.langchain.com/langsmith/evaluators#create-an-evaluator-in-the-ui)

### Steps

1. Open LangSmith → **Evaluators** → **+ Evaluator**
2. Choose **Create from a template**, then a **Quality** template for correctness / factual grounding / hallucination (exact names vary by workspace)
3. Rename it to something specific, e.g. `research-grounding-quality`
4. Edit the rubric so it encodes this firm's research standard. A workable starting point:

```text
Score 1 only if ALL of the following hold for the assistant's answer:
- every number, name, and date appears in the retrieved filing/transcript evidence
- at least one specific source is attributed (e.g. "[source: NBRI 10-Q FY26 Q2]")
- the most recent retrieved document wins when documents conflict
- the answer states what is unknown instead of predicting undisclosed outcomes
- no personalized buy/sell/allocation advice and no offer to place trades

Score 0 if any claim is unsupported, contradicted, stale, or uncited.
```

5. Set **variable mapping** to the flattened fields (these exist precisely so template variables map cleanly):

| Template variable  | Map to                    |
| ------------------ | ------------------------- |
| `input`            | `inputs.question`         |
| `output`           | `outputs.answer`          |
| `reference`        | `reference_outputs.answer` |

  Map to `outputs.messages` instead if you want the judge to see retrieved tool
  evidence — worth demoing, since grounding is hard to judge from the answer alone.
  Remove the `reference` variable if you want a reference-free judge.

6. Pick a low-temperature judge model and a binary score
7. Attach the evaluator to the `research-quality` dataset (or run it on the seeded experiment) and run it
8. Compare the judge's score against the human label in `reference_outputs.expected_score`, and read `reference_outputs.label_rationale` on any row where they disagree

### Where the interesting disagreements are

Group the experiment by the `category` run metadata to find them fast:

| Category                          | Rows | What it stresses in the rubric                                    |
| --------------------------------- | ---- | ----------------------------------------------------------------- |
| `well_grounded`                   | 5    | Should score 1 — baseline sanity check                            |
| `missing_citations`               | 4    | Facts are correct but uncited; templates often score these 1       |
| `unsupported_claims`              | 3    | Numbers absent from or contradicting the retrieved evidence        |
| `stale_info_presented_as_current` | 2    | Cited, accurate-in-isolation, superseded by a newer document       |
| `uncertainty_handled_well`        | 3    | Should score 1 — punish rubrics that treat "I can't verify" as failure |
| `personalized_recommendation`     | 3    | Advice-boundary violations a grounding-only rubric will miss       |

### Talking points

- Templates get you most of the way; the firm-specific standard lives in the rubric edits
- `missing_citations` and `stale_info_presented_as_current` are where a stock correctness template usually disagrees with the human label — good motivation for Exercises 2 and 3
- Reference-free templates carry over to online monitoring; reference-based ones need the labeled `expected_outputs` this repo seeds
- Keep the first customization small so experiment diffs stay interpretable

---



## Exercise 2 — Custom code evaluators

**Goal:** Show deterministic policy checks that do not need an LLM.

Implementation: `[evaluators/research_policy.py](evaluators/research_policy.py)`


| Evaluator                          | Passes when                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `citation_coverage_evaluator`      | Final AI answer includes a source citation marker           |
| `no_personalized_advice_evaluator` | Answer avoids buy/sell/allocate-style personalized language |




### Option A — paste into LangSmith UI

1. Dataset / experiment → add **Code** evaluator
2. Paste a minimal function (UI online style receives `run` only), for example:

```python
import re

def perform_eval(run):
    outputs = run.get("outputs") or {}
    messages = outputs.get("messages") or []
    ai_texts = [
        m.get("content", "")
        for m in messages
        if isinstance(m, dict) and m.get("type") == "ai" and m.get("content")
    ]
    text = "\n".join(ai_texts).lower()
    has_citation = bool(
        re.search(r"\[source:|\(source:|\baccording to\b|\bper (the )?(10-[kq]|filing|report|transcript)\b", text)
    )
    prohibited = bool(
        re.search(
            r"\byou should (buy|sell|hold)\b|\bi recommend (buying|selling|holding)\b|\ballocate\s+\d+%\b|\brebalance your portfolio\b|\bplace an? (buy|sell) order\b",
            text,
        )
    )
    return {
        "has_citation": has_citation,
        "no_personalized_advice": not prohibited,
    }
```

1. Run on `research-quality` and show `personalized_recommendation` / `missing_citations` rows flipping red/green



### Option B — local SDK evaluate (optional follow-up)

Use `citation_coverage_evaluator` / `no_personalized_advice_evaluator` with `langsmith.evaluate` against the uploaded dataset once an agent target function exists. This repo seeds **synthetic traces** so the UI path is enough for the live demo.

---



## Exercise 3 — Align Evaluator (calibrate LLM-as-judge)

**Goal:** Show how to calibrate escalation judgment with human experts.

Follow the full facilitator script:

→ `[docs/align_evaluator_walkthrough.md](docs/align_evaluator_walkthrough.md)`

Starter prompt (intentionally incomplete):

→ `[judge_prompts/escalation_accuracy.md](judge_prompts/escalation_accuracy.md)`

Example refined prompt after alignment:

→ `[judge_prompts/escalation_accuracy_aligned.md](judge_prompts/escalation_accuracy_aligned.md)`

Short version:

1. Attach an LLM judge using the starter prompt to `escalation-accuracy`
2. Send a mixed sample of experiment runs to an **annotation queue**
3. Collect expert labels
4. Use **Align Evaluator** to find disagreements
5. Tighten the rubric (MNPI, over-escalation, false claims)
6. Re-test agreement; optionally enable few-shot corrections

---



## Suggested 30-minute agenda


| Minutes | Segment                                                           |
| ------- | ----------------------------------------------------------------- |
| 0–5     | Context: research assistant + eval surfaces in LangSmith          |
| 5–12    | Exercise 1: customize a template evaluator on `research-quality`  |
| 12–20   | Exercise 2: code evaluator for citations + no personalized advice |
| 20–28   | Exercise 3: Align Evaluator on `escalation-accuracy`              |
| 28–30   | Recap: template → custom → human-aligned judge                    |




## Safety notes

- Synthetic data only — do not paste real client, employee, or deal information into demos
- `.env` is gitignored; never commit API keys
- Code evaluators run without network access in LangSmith — keep imports to the standard library (or the documented allowlist)

