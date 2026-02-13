# Objectives-to-Patterns Anki Generator

Generate Anki flashcards from medical learning objectives using the Claude Batches API. Takes a spreadsheet of Step 2 CK objectives and produces pattern-recognition cards ready for Anki import.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Input format

A TSV file (e.g. exported from Google Sheets) with at least an `Objective` column. Optional columns for auto-tagging:

| Objective | Subject | System | Topic |
|---|---|---|---|
| Diagnose and manage acute cholecystitis in pregnancy | OB/GYN | GI | Biliary |

Cards will be tagged like `OB/GYN::GI::Biliary` in the output.

## Usage

```bash
python generate_anki_cards.py input.tsv output.tsv
```

### Options

| Flag | Description |
|---|---|
| `--density N` | Cards per objective (default: 2.0). See below. |
| `--test` | Only process the first 50 objectives (for testing). |
| `--dry-run` | Show estimated card count, token usage, and cost without calling the API. |
| `--batch-size N` | Split into multiple batches of N requests each. |

### Density

Controls how many cards Claude generates per objective. Accepts any float value.

| Value | Behavior |
|---|---|
| `0.25` | Very selective — only ~25% of objectives get a card, rest are skipped as low-yield |
| `0.5` | ~50% of objectives get a card |
| `0.75` | ~75% of objectives get a card |
| `1` | Exactly 1 card per objective |
| `2` | Up to 2 cards per objective (default) |
| `3` | Up to 3 cards, covering distinct testable angles |

```bash
# Slim deck — only the highest-yield patterns
python generate_anki_cards.py input.tsv output.tsv --density 0.5

# One card per objective
python generate_anki_cards.py input.tsv output.tsv --density 1

# Broader coverage
python generate_anki_cards.py input.tsv output.tsv --density 3
```

### Dry run

Check token usage and cost before committing to an API call:

```bash
python generate_anki_cards.py input.tsv output.tsv --dry-run --density 1
```

```
--- Dry Run Summary ---
Objectives to process: 500
Batches: 1
Density: 1.0 (est. ~1.0 cards/objective)
Estimated cards: 500
Estimated tokens: ~350,600 (300,600 in + 50,000 out)
Estimated cost (Batch API): $0.45
Model: claude-sonnet-4-20250514
```

Token estimates are useful for checking costs before committing to an API call.

## Output format

A TSV file with three columns, ready for Anki import:

```
Front	Back	Tags
Woman age <45 with abnormal uterine bleeding → first step?	Pregnancy test (rule out pregnancy before further evaluation)	OB/GYN::Reproductive::AUB
```

Cards follow a pattern-recognition format: clinical presentation + key findings → question type, with concise answers including distinguishing rationale.

## Resume support

If the script is interrupted mid-run, it saves state to `output.tsv.state.json`. Re-running the same command will pick up where it left off and poll the existing batches instead of resubmitting.

## How it works

1. Reads objectives from the input TSV
2. Submits them to the Claude Batches API (one request per objective, system prompt cached)
3. Polls until the batch completes
4. Parses TSV card output from each response
5. Retries any failed requests once
6. Deduplicates cards with similar front text
7. Writes the final TSV

## Disclaimer

This is a vibe-coded project. The code was generated almost entirely by [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic's AI coding agent) and only lightly reviewed by the author. AI-generated flashcards may contain inaccuracies — review all output cards for medical accuracy before using them for study. Use at your own discretion.

## Acknowledgments

The learning objectives used as input come from the incredible **Step 2 CK UWorld Educational Objectives** spreadsheet compiled by [u/tamsulosinflomax](https://www.reddit.com/r/Step2/comments/j64mnn/i_did_a_thing_step_2_ck_uworld_educational/). Thank you for putting in the work to organize all of these.

- [Reddit post](https://www.reddit.com/r/Step2/comments/j64mnn/i_did_a_thing_step_2_ck_uworld_educational/)
- [Google Sheets](https://docs.google.com/spreadsheets/d/1Zitsgfy2NJrB9kTEQwF6FXkICOgqJsLV-1sv6hKscB8/edit?gid=704422695#gid=704422695)
