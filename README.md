# Objectives-to-Patterns Anki Generator

Generate Anki flashcards from medical learning objectives using batch APIs. Supports Claude, OpenAI, and Gemini. Takes a spreadsheet of Step 2 CK objectives and produces pattern-recognition cards ready for Anki import.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API key

You need a pay-as-you-go API key from whichever provider you want to use. Pick one:

| Provider | Get a key at | Env var |
|----------|-------------|---------|
| Claude | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` |
| OpenAI | [platform.openai.com](https://platform.openai.com) | `OPENAI_API_KEY` |
| Gemini | [aistudio.google.com](https://aistudio.google.com) | `GEMINI_API_KEY` |

Three ways to provide your key (in priority order):

1. **`--api-key` flag** — pass directly on the command line
2. **`.env` file** — create a `.env` file in the project root (requires `python-dotenv`)
3. **Environment variable** — `export OPENAI_API_KEY="sk-..."`

Example `.env` file:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
ANTHROPIC_API_KEY=sk-ant-...
```

## Input format

A TSV file (e.g. exported from Google Sheets) with at least an `Objective` column. Optional columns for auto-tagging:

| Objective | Subject | System | Topic |
|---|---|---|---|
| Diagnose and manage acute cholecystitis in pregnancy | OB/GYN | GI | Biliary |

Cards will be tagged like `OB/GYN::GI::Biliary` in the output.

## Usage

```bash
# Default: Claude Sonnet (~$7 for full dataset)
python generate_anki_cards.py input.tsv output.tsv

# Comparable quality: GPT-4.1 (~$3.50) or Gemini 2.5 Flash (~$1)
python generate_anki_cards.py input.tsv output.tsv --provider openai
python generate_anki_cards.py input.tsv output.tsv --provider gemini

# Budget options: use cheaper models with --model flag
python generate_anki_cards.py input.tsv output.tsv --provider openai --model gpt-4.1-mini
python generate_anki_cards.py input.tsv output.tsv --provider gemini --model gemini-2.5-flash-lite
```

### Options

| Flag | Description |
|---|---|
| `--provider {claude,openai,gemini}` | API provider (default: claude). |
| `--model MODEL` | Override the default model for the chosen provider. |
| `--api-key KEY` | API key (alternative to env var / .env file). |
| `--density N` | Cards per objective (default: 2.0). See below. |
| `--test` | Only process the first 50 objectives (for testing). |
| `--dry-run` | Show estimated card count, token usage, and cost without calling the API. |
| `--batch-size N` | Split into multiple batches of N requests each. |

### Provider defaults and estimated costs

All cost estimates are for ~4,100 objectives at density 2.0 using batch APIs (50% discount already applied).

| Provider | Default model | Quality tier | Est. cost | Notes |
|----------|--------------|--------------|-----------|-------|
| `claude` | claude-sonnet-4-20250514 | Premium | ~$7.00 | Highest quality, prompt caching |
| `openai` | gpt-4.1 | Premium | ~$3.50 | Comparable to Sonnet, 50% cheaper |
| `gemini` | gemini-2.5-flash | Premium | ~$1.00 | Comparable to Sonnet, 85% cheaper |

### Available models by provider

**Recommendation:** Start with the defaults above. If quality is acceptable but you want to save costs, try the "Budget" tier models with `--test` first.

#### Claude
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `claude-sonnet-4-20250514` | Premium | $1.50 | $7.50 | Default, prompt caching enabled |

#### OpenAI
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `gpt-4.1` | Premium | $1.00 | $4.00 | Default, comparable to Sonnet |
| `gpt-4o` | Premium | $1.25 | $5.00 | Previous-gen flagship |
| `gpt-4.1-mini` | Budget | $0.20 | $0.80 | ~90% cheaper, good for simple tasks |
| `gpt-4o-mini` | Budget | $0.075 | $0.30 | ~95% cheaper, good for simple tasks |
| `gpt-4.1-nano` | Budget | $0.05 | $0.20 | Cheapest, may sacrifice quality |

#### Gemini
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `gemini-2.5-flash` | Premium | $0.15 | $1.25 | Default, comparable to Sonnet |
| `gemini-2.5-pro` | Premium | $0.625 | $5.00 | Highest quality Gemini model |
| `gemini-2.5-flash-lite` | Budget | $0.05 | $0.20 | ~95% cheaper, may sacrifice quality |
| `gemini-2.0-flash` | Budget | $0.05 | $0.20 | Previous-gen, may sacrifice quality |

### Density

Controls how many cards are generated per objective. Accepts any float value.

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
python generate_anki_cards.py input.tsv output.tsv --dry-run --provider openai
```

```
--- Dry Run Summary ---
Provider: openai
Model: gpt-4o-mini
Objectives to process: 500
Batches: 1
Density: 2.0 (est. ~2.0 cards/objective)
Estimated tokens: ~400,000 (300,000 in + 100,000 out)
Estimated cost (Batch API): $0.05
```

## Output format

A TSV file with three columns, ready for Anki import:

```
Front	Back	Tags
Woman age <45 with abnormal uterine bleeding → first step?	Pregnancy test (rule out pregnancy before further evaluation)	OB/GYN::Reproductive::AUB
```

Cards follow a pattern-recognition format: clinical presentation + key findings → question type, with concise answers including distinguishing rationale.

## Resume support

If the script is interrupted mid-run, it saves state to `output.tsv.state.json`. Re-running the same command will pick up where it left off and poll the existing batches instead of resubmitting. The state file records which provider was used, so resume always uses the correct API.

## How it works

1. Reads objectives from the input TSV
2. Submits them to the chosen provider's batch API (one request per objective)
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
