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
python generate_anki_cards.py input.tsv output.tsv --provider claude --model claude-haiku-4-5-20251001
python generate_anki_cards.py input.tsv output.tsv --provider openai --model gpt-4.1-mini
python generate_anki_cards.py input.tsv output.tsv --provider gemini --model gemini-2.5-flash-lite

# Premium option: Claude Opus 4.6 for highest quality (~$15 for full dataset)
python generate_anki_cards.py input.tsv output.tsv --provider claude --model claude-opus-4-6
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
| `--filter-subject SUBJECTS` | Filter by Subject (comma-separated, case-insensitive). |
| `--filter-system SYSTEMS` | Filter by System (comma-separated, case-insensitive). |
| `--filter-topic TOPICS` | Filter by Topic (comma-separated, case-insensitive). |
| `--fuzzy` | Enable fuzzy matching for filters (allows abbreviations and typos). |

### Provider defaults and estimated costs

All cost estimates are for ~4,100 objectives at density 2.0 using batch APIs (50% discount already applied).

| Provider | Default model | Quality tier | Est. cost | Notes |
|----------|--------------|--------------|-----------|-------|
| `claude` | claude-sonnet-4-5-20250929 | Premium | ~$7.00 | Latest Sonnet, prompt caching |
| `openai` | gpt-4.1 | Premium | ~$3.50 | Comparable to Sonnet, 50% cheaper |
| `gemini` | gemini-2.5-flash | Premium | ~$1.00 | Comparable to Sonnet, 85% cheaper |

**Claude model comparison for full dataset:**
- Opus 4.6 (~$15): Highest quality available
- Sonnet 4.5 (~$7): Default, excellent quality/cost balance
- Haiku 4.5 (~$3): Budget option, 60% cheaper than Sonnet

### Available models by provider

**Recommendation:** Start with the defaults above. If quality is acceptable but you want to save costs, try the "Budget" tier models with `--test` first.

**Note:** All providers use Batch APIs with 50% discount (already reflected in pricing). Batch processing completes within 24 hours, ideal for non-urgent workloads.

#### Claude
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `claude-opus-4-6` | Premium+ | $2.50 | $12.50 | Batch API + prompt caching (Feb 2026) |
| `claude-sonnet-4-5-20250929` | Premium | $1.50 | $7.50 | Default, Batch API + prompt caching |
| `claude-haiku-4-5-20251001` | Budget | $0.50 | $2.50 | Batch API + prompt caching |

#### OpenAI
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `gpt-4.1` | Premium | $1.00 | $4.00 | Default, Batch API (50% off) |
| `gpt-4o` | Premium | $1.25 | $5.00 | Batch API (50% off) |
| `gpt-4.1-mini` | Budget | $0.20 | $0.80 | Batch API (50% off) |
| `gpt-4o-mini` | Budget | $0.075 | $0.30 | Batch API (50% off) |
| `gpt-4.1-nano` | Budget | $0.05 | $0.20 | Batch API (50% off) |

#### Gemini
| Model | Tier | Input $/M | Output $/M | Notes |
|-------|------|-----------|------------|-------|
| `gemini-2.5-flash` | Premium | $0.15 | $1.25 | Default, Batch API (50% off) |
| `gemini-2.5-pro` | Premium | $0.625 | $5.00 | Batch API (50% off) |
| `gemini-2.5-flash-lite` | Budget | $0.05 | $0.20 | Batch API (50% off) |
| `gemini-2.0-flash` | Budget | $0.05 | $0.20 | Batch API (50% off) |

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

### Filtering objectives

Filter which objectives to process by Subject, System, or Topic. Use comma-separated values for multiple selections (case-insensitive).

By default, filtering uses **exact matching** (case-insensitive). Add the `--fuzzy` flag to enable fuzzy matching, which allows abbreviations and minor typos (e.g., "obgyn" matches "Obstetrics & Gynecology", "cardio" matches "Cardiovascular System").

**Note:** Fuzzy matching uses common medical abbreviations and string similarity. While it handles most common cases well, it may occasionally produce unexpected matches. Always check the output to see what was matched (shown as `→ 'filter' matched X (fuzzy): ...`).

#### Valid filter values

**Subjects** (all 5 subjects):
- `Medicine`
- `Obstetrics & Gynecology`
- `Pediatrics`
- `Psychiatry`
- `Surgery`

**Systems** (22 total; examples shown):
- `Cardiovascular System`
- `Pulmonary & Critical Care`
- `Gastrointestinal & Nutrition`
- `Female Reproductive System & Breast`
- `Nervous System`
- `Endocrine, Diabetes & Metabolism`
- ...and 16 more

**Topics** (1,200+ total; examples shown):
- `Abnormal Uterine Bleeding`
- `Acute Kidney Injury`
- `Atrial Fibrillation`
- `Diabetes Mellitus`
- `Myocardial Infarction`
- `Pneumonia`
- ...and 1,200+ more

For the complete list of all valid subjects, systems, and topics, see [FILTRATION_VALUES.md](FILTRATION_VALUES.md).

#### Usage examples

```bash
# Filter by exact Subject name
python generate_anki_cards.py input.tsv output.tsv --filter-subject "Obstetrics & Gynecology,Pediatrics"

# Filter by exact System name
python generate_anki_cards.py input.tsv output.tsv --filter-system "Cardiovascular System"

# Use fuzzy matching for abbreviations (recommended for convenience)
python generate_anki_cards.py input.tsv output.tsv --filter-subject "obgyn" --fuzzy
python generate_anki_cards.py input.tsv output.tsv --filter-system "cardio" --fuzzy
python generate_anki_cards.py input.tsv output.tsv --filter-subject "psych" --fuzzy

# Combine multiple filters (all conditions must match)
python generate_anki_cards.py input.tsv output.tsv \
  --filter-subject "Medicine" \
  --filter-system "Pulmonary & Critical Care" \
  --fuzzy

# Dry run with filtering to see what matches
python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "cardio" \
  --fuzzy
```

When filtering, the dry run will show you:
- Total objectives in the file
- How many match your filters
- How many were filtered out
- Estimated cost for just the filtered objectives

### Dry run

Check token usage and cost before committing to an API call:

```bash
python generate_anki_cards.py input.tsv output.tsv --dry-run --provider openai
```

```
--- Dry Run Summary ---
Provider: openai
Model: gpt-4o-mini
Total objectives in file: 4100
Objectives to process: 500 (no filters applied)
Batches: 1
Density: 2.0 (est. ~2.0 cards/objective)
Estimated tokens: ~400,000 (300,000 in + 100,000 out)
Estimated cost (Batch API): $0.05
```

### Cost confirmation

For real runs (not `--dry-run`), if the estimated cost exceeds **$0.10**, the script will show a confirmation prompt before proceeding:

```
--- Cost Confirmation ---
Estimated cost: $3.50
Objectives: 4100
Estimated tokens: ~2,500,000
Continue with this run? [y/N]:
```

This prevents accidentally running expensive batches. Type `y` or `yes` to continue, or any other key to cancel.

## Common workflows

### Process one subject at a time

```bash
# Day 1: OB/GYN
python generate_anki_cards.py input.tsv cards_obgyn.tsv \
  --filter-subject "Obstetrics & Gynecology"

# Day 2: Pediatrics
python generate_anki_cards.py input.tsv cards_peds.tsv \
  --filter-subject "Pediatrics"

# Day 3: Medicine
python generate_anki_cards.py input.tsv cards_medicine.tsv \
  --filter-subject "Medicine"
```

### Focus on specific systems

```bash
# Just cardiovascular across all subjects
python generate_anki_cards.py input.tsv cards_cardio.tsv \
  --filter-system "Cardiovascular"

# Just pulmonary
python generate_anki_cards.py input.tsv cards_pulm.tsv \
  --filter-system "Pulmonary & Critical Care"
```

### Test with small subset before full run

```bash
# Step 1: Dry run to see the scope
python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Obstetrics & Gynecology"

# Step 2: Test with first 50 matching objectives
python generate_anki_cards.py input.tsv output_test.tsv --test \
  --filter-subject "Obstetrics & Gynecology"

# Step 3: If quality looks good, run the full set
python generate_anki_cards.py input.tsv output_full.tsv \
  --filter-subject "Obstetrics & Gynecology"
```

### Adjust density based on scope

```bash
# High-yield only (50% of objectives) for a large subject
python generate_anki_cards.py input.tsv cards_medicine_selective.tsv \
  --filter-subject "Medicine" \
  --density 0.5

# Comprehensive coverage (3 cards per objective) for a small topic
python generate_anki_cards.py input.tsv cards_aub_comprehensive.tsv \
  --filter-topic "Abnormal Uterine Bleeding" \
  --density 3
```

### Compare providers and models

```bash
# Compare Claude models
python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --provider claude --model claude-haiku-4-5-20251001

python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --provider claude --model claude-sonnet-4-5-20250929

python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --provider claude --model claude-opus-4-6

# Try different providers with same filter
python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --filter-system "Pulmonary & Critical Care" \
  --provider openai

python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --filter-system "Pulmonary & Critical Care" \
  --provider gemini

# Test cheaper models across providers
python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --provider claude --model claude-haiku-4-5-20251001

python generate_anki_cards.py input.tsv output.tsv --dry-run \
  --filter-subject "Medicine" \
  --provider openai --model gpt-4.1-mini
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
