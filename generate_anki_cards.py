#!/usr/bin/env python3
"""Generate Anki flashcards from medical learning objectives using batch APIs (Claude, OpenAI, Gemini)."""

import abc
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

MAX_TOKENS = 1024
POLL_INTERVAL_SECONDS = 60
DEFAULT_TEST_LIMIT = 50
DEFAULT_DENSITY = 2.0

SYSTEM_PROMPT_TEMPLATE = """You are creating Anki flashcards for medical education (Step 2 CK prep). Given a learning objective, {density_instruction} in this format:

Front: [Clinical presentation] + [key findings] → [question type]?
Back: [Answer] ([brief rationale or distinguishing feature])

Example:
Front: Woman age <45 with abnormal uterine bleeding → first step?
Back: Pregnancy test (rule out pregnancy before further evaluation)

Example:
Front: Adolescent with hypothalamic-pituitary axis dysfunction + anovulatory cycles → treatment?
Back: Progesterone withdrawal (induce withdrawal bleed, confirm responsive endometrium)

Example:
Front: Pregnant patient with RUQ pain + leukocytosis + elevated aminotransferases → diagnosis?
Back: Acute fatty liver of pregnancy (high maternal/fetal mortality - requires immediate delivery)

Rules:
- Focus on clinical decision-making patterns, not isolated facts
- Include enough context to make the answer unambiguous
- Keep answers concise but complete with key distinguishing features
- Only high-yield, testable material
- Avoid redundant cards testing the same concept
- Return cards in TSV format: Front[TAB]Back with one card per line
- Do NOT include the header row "Front\\tBack" in your response
- If the objective doesn't lend itself to a good clinical pattern card, return an empty response"""

# Batch API pricing per million tokens (already includes 50% batch discount)
PRICING = {
    # Claude
    "claude-sonnet-4-20250514": {"input": 1.50, "output": 7.50, "cached_input": 0.15},
    # OpenAI
    "gpt-4.1": {"input": 1.00, "output": 4.00},
    "gpt-4.1-mini": {"input": 0.20, "output": 0.80},
    "gpt-4.1-nano": {"input": 0.05, "output": 0.20},
    "gpt-4o": {"input": 1.25, "output": 5.00},
    "gpt-4o-mini": {"input": 0.075, "output": 0.30},
    # Gemini
    "gemini-2.5-flash": {"input": 0.15, "output": 1.25},
    "gemini-2.5-pro": {"input": 0.625, "output": 5.00},
    "gemini-2.5-flash-lite": {"input": 0.05, "output": 0.20},
    "gemini-2.0-flash": {"input": 0.05, "output": 0.20},
}


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class BatchProvider(abc.ABC):
    name: str
    default_model: str
    env_var: str

    @abc.abstractmethod
    def create_client(self, api_key):
        ...

    @abc.abstractmethod
    def submit_batch(self, client, rows, system_prompt, model, batch_size):
        """Submit batch and return list of batch/job identifiers."""
        ...

    @abc.abstractmethod
    def poll_batch(self, client, batch_ids):
        """Block until all batches finish. Returns list of completed batch objects."""
        ...

    @abc.abstractmethod
    def retrieve_results(self, client, batch_ids, rows_by_custom_id):
        """Return (cards, failed_ids)."""
        ...

    def estimate_cost(self, num_objectives, density, model):
        prices = PRICING.get(model)
        if not prices:
            return None

        system_tokens = 500
        input_per_request = 100
        output_per_request = int(200 * max(density, 0.5) / DEFAULT_DENSITY)

        total_input_tokens = num_objectives * (system_tokens + input_per_request)
        total_output_tokens = num_objectives * output_per_request

        # Claude has prompt caching — first request pays full, rest cached
        if "cached_input" in prices:
            full_input_tokens = system_tokens + input_per_request
            cached_input_tokens = input_per_request
            input_cost = (full_input_tokens * prices["input"] / 1_000_000) + (
                (num_objectives - 1) * cached_input_tokens * prices["input"] / 1_000_000
            )
            cached_cost = (num_objectives - 1) * system_tokens * prices["cached_input"] / 1_000_000
        else:
            input_cost = total_input_tokens * prices["input"] / 1_000_000
            cached_cost = 0

        output_cost = total_output_tokens * prices["output"] / 1_000_000

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost": input_cost + cached_cost + output_cost,
        }


class ClaudeProvider(BatchProvider):
    name = "claude"
    default_model = "claude-sonnet-4-20250514"
    env_var = "ANTHROPIC_API_KEY"

    def create_client(self, api_key):
        import anthropic
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def submit_batch(self, client, rows, system_prompt, model, batch_size):
        import anthropic

        requests = []
        for row in rows:
            requests.append({
                "custom_id": f"obj_{row['index']}",
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [
                        {"role": "user", "content": row["objective"]},
                    ],
                },
            })

        chunks = list(chunk_list(requests, batch_size)) if batch_size and batch_size < len(requests) else [requests]
        batch_ids = []

        for i, chunk in enumerate(chunks):
            print(f"Submitting batch {i + 1}/{len(chunks)} ({len(chunk)} requests)...", file=sys.stderr)
            try:
                batch = client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"  Batch ID: {batch.id}", file=sys.stderr)
            except anthropic.APIError as e:
                print(f"  Error submitting batch {i + 1}: {e}", file=sys.stderr)
                time.sleep(10)
                try:
                    batch = client.messages.batches.create(requests=chunk)
                    batch_ids.append(batch.id)
                    print(f"  Batch ID (retry): {batch.id}", file=sys.stderr)
                except anthropic.APIError as e2:
                    print(f"  Failed after retry: {e2}", file=sys.stderr)

        return batch_ids

    def poll_batch(self, client, batch_ids):
        pending = set(batch_ids)
        finished = []

        while pending:
            for batch_id in list(pending):
                batch = client.messages.batches.retrieve(batch_id)
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                print(
                    f"  Batch {batch_id[:16]}... — status: {batch.processing_status}, "
                    f"succeeded: {counts.succeeded}/{total}",
                    file=sys.stderr,
                )
                if batch.processing_status == "ended":
                    pending.discard(batch_id)
                    finished.append(batch)

            if pending:
                print(f"Waiting {POLL_INTERVAL_SECONDS}s before next poll ({len(pending)} batch(es) remaining)...", file=sys.stderr)
                time.sleep(POLL_INTERVAL_SECONDS)

        return finished

    def retrieve_results(self, client, batch_ids, rows_by_custom_id):
        cards = []
        failed_ids = []

        for batch_id in batch_ids:
            for result in client.messages.batches.results(batch_id):
                custom_id = result.custom_id
                row = rows_by_custom_id.get(custom_id)
                tag = build_tag(row) if row else ""

                if result.result.type == "succeeded":
                    text = ""
                    for block in result.result.message.content:
                        if block.type == "text":
                            text += block.text
                    cards.extend(parse_card_lines(text, tag, custom_id))
                else:
                    failed_ids.append(custom_id)
                    print(f"  {custom_id} failed: {result.result.type}", file=sys.stderr)

        return cards, failed_ids


class OpenAIProvider(BatchProvider):
    name = "openai"
    default_model = "gpt-4.1"
    env_var = "OPENAI_API_KEY"

    def create_client(self, api_key):
        from openai import OpenAI
        return OpenAI(api_key=api_key) if api_key else OpenAI()

    def submit_batch(self, client, rows, system_prompt, model, batch_size):
        chunks = list(chunk_list(rows, batch_size)) if batch_size and batch_size < len(rows) else [rows]
        batch_ids = []

        for i, chunk in enumerate(chunks):
            print(f"Submitting batch {i + 1}/{len(chunks)} ({len(chunk)} requests)...", file=sys.stderr)

            # Write JSONL to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for row in chunk:
                    line = {
                        "custom_id": f"obj_{row['index']}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "max_tokens": MAX_TOKENS,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": row["objective"]},
                            ],
                        },
                    }
                    f.write(json.dumps(line) + "\n")
                tmp_path = f.name

            try:
                batch_file = client.files.create(
                    file=open(tmp_path, "rb"),
                    purpose="batch",
                )
                batch = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                batch_ids.append(batch.id)
                print(f"  Batch ID: {batch.id}", file=sys.stderr)
            except Exception as e:
                print(f"  Error submitting batch {i + 1}: {e}", file=sys.stderr)
                time.sleep(10)
                try:
                    batch_file = client.files.create(
                        file=open(tmp_path, "rb"),
                        purpose="batch",
                    )
                    batch = client.batches.create(
                        input_file_id=batch_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                    )
                    batch_ids.append(batch.id)
                    print(f"  Batch ID (retry): {batch.id}", file=sys.stderr)
                except Exception as e2:
                    print(f"  Failed after retry: {e2}", file=sys.stderr)
            finally:
                os.unlink(tmp_path)

        return batch_ids

    def poll_batch(self, client, batch_ids):
        pending = set(batch_ids)
        finished = []

        while pending:
            for batch_id in list(pending):
                batch = client.batches.retrieve(batch_id)
                counts = batch.request_counts
                print(
                    f"  Batch {batch_id[:16]}... — status: {batch.status}, "
                    f"completed: {counts.completed}/{counts.total}",
                    file=sys.stderr,
                )
                if batch.status in ("completed", "failed", "expired", "cancelled"):
                    pending.discard(batch_id)
                    finished.append(batch)

            if pending:
                print(f"Waiting {POLL_INTERVAL_SECONDS}s before next poll ({len(pending)} batch(es) remaining)...", file=sys.stderr)
                time.sleep(POLL_INTERVAL_SECONDS)

        return finished

    def retrieve_results(self, client, batch_ids, rows_by_custom_id):
        cards = []
        failed_ids = []

        for batch_id in batch_ids:
            batch = client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                print(f"  Batch {batch_id[:16]}... has no output file (status: {batch.status})", file=sys.stderr)
                continue

            content = client.files.content(batch.output_file_id).content
            for line in content.decode("utf-8").strip().splitlines():
                result = json.loads(line)
                custom_id = result["custom_id"]
                row = rows_by_custom_id.get(custom_id)
                tag = build_tag(row) if row else ""

                response = result.get("response", {})
                if response.get("status_code") == 200:
                    text = response["body"]["choices"][0]["message"]["content"]
                    cards.extend(parse_card_lines(text, tag, custom_id))
                else:
                    failed_ids.append(custom_id)
                    error = result.get("error", {})
                    print(f"  {custom_id} failed: {error.get('message', 'unknown error')}", file=sys.stderr)

        return cards, failed_ids


class GeminiProvider(BatchProvider):
    name = "gemini"
    default_model = "gemini-2.5-flash"
    env_var = "GEMINI_API_KEY"

    def create_client(self, api_key):
        from google import genai
        return genai.Client(api_key=api_key) if api_key else genai.Client()

    def submit_batch(self, client, rows, system_prompt, model, batch_size):
        chunks = list(chunk_list(rows, batch_size)) if batch_size and batch_size < len(rows) else [rows]
        batch_ids = []

        for i, chunk in enumerate(chunks):
            print(f"Submitting batch {i + 1}/{len(chunks)} ({len(chunk)} requests)...", file=sys.stderr)

            inline_requests = []
            for row in chunk:
                inline_requests.append({
                    "contents": [
                        {"parts": [{"text": row["objective"]}], "role": "user"},
                    ],
                    "config": {
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "max_output_tokens": MAX_TOKENS,
                    },
                })

            try:
                batch_job = client.batches.create(
                    model=f"models/{model}",
                    src=inline_requests,
                    config={"display_name": f"anki-batch-{i + 1}"},
                )
                batch_ids.append(batch_job.name)
                print(f"  Batch ID: {batch_job.name}", file=sys.stderr)
            except Exception as e:
                print(f"  Error submitting batch {i + 1}: {e}", file=sys.stderr)
                time.sleep(10)
                try:
                    batch_job = client.batches.create(
                        model=f"models/{model}",
                        src=inline_requests,
                        config={"display_name": f"anki-batch-{i + 1}"},
                    )
                    batch_ids.append(batch_job.name)
                    print(f"  Batch ID (retry): {batch_job.name}", file=sys.stderr)
                except Exception as e2:
                    print(f"  Failed after retry: {e2}", file=sys.stderr)

        return batch_ids

    def poll_batch(self, client, batch_ids):
        completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}
        pending = set(batch_ids)
        finished = []

        while pending:
            for job_name in list(pending):
                job = client.batches.get(name=job_name)
                state = job.state if isinstance(job.state, str) else job.state.name
                short_name = job_name.split("/")[-1][:16] if "/" in job_name else job_name[:16]
                print(f"  Batch {short_name}... — state: {state}", file=sys.stderr)
                if state in completed_states:
                    pending.discard(job_name)
                    finished.append(job)

            if pending:
                print(f"Waiting {POLL_INTERVAL_SECONDS}s before next poll ({len(pending)} batch(es) remaining)...", file=sys.stderr)
                time.sleep(POLL_INTERVAL_SECONDS)

        return finished

    def retrieve_results(self, client, batch_ids, rows_by_custom_id):
        cards = []
        failed_ids = []

        # Gemini inline responses are indexed by position, so we need the original row order
        custom_id_list = sorted(rows_by_custom_id.keys(), key=lambda cid: int(cid.split("_")[1]))

        for job_name in batch_ids:
            job = client.batches.get(name=job_name)
            state = job.state if isinstance(job.state, str) else job.state.name

            if state != "JOB_STATE_SUCCEEDED":
                print(f"  Batch {job_name} ended with state: {state}", file=sys.stderr)
                failed_ids.extend(custom_id_list)
                continue

            if not hasattr(job, "dest") or not hasattr(job.dest, "inlined_responses"):
                print(f"  Batch {job_name} has no inline responses", file=sys.stderr)
                failed_ids.extend(custom_id_list)
                continue

            for idx, inline_response in enumerate(job.dest.inlined_responses):
                custom_id = custom_id_list[idx] if idx < len(custom_id_list) else f"obj_{idx}"
                row = rows_by_custom_id.get(custom_id)
                tag = build_tag(row) if row else ""

                try:
                    text = inline_response.response.text
                    cards.extend(parse_card_lines(text, tag, custom_id))
                except Exception:
                    failed_ids.append(custom_id)
                    print(f"  {custom_id} failed: could not extract text", file=sys.stderr)

        return cards, failed_ids


PROVIDERS = {
    "claude": ClaudeProvider(),
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_density_instruction(density):
    """Return the prompt instruction fragment for the given card density."""
    if density < 1.0:
        pct = int(density * 100)
        return (
            f"create at most 1 card. Target roughly {pct}% of objectives — "
            "skip objectives that are low-yield for Step 2 CK by returning an empty response. "
            "Only keep the most important clinical decision-making patterns"
        )
    elif density == 1.0:
        return "create exactly 1 concise pattern recognition card"
    else:
        n = int(round(density))
        return f"create up to {n} pattern recognition cards, covering distinct testable angles"


def build_system_prompt(density):
    """Build the system prompt with the appropriate density instruction."""
    return SYSTEM_PROMPT_TEMPLATE.format(density_instruction=get_density_instruction(density))


def read_objectives(path):
    """Read TSV file and return list of row dicts with Objective and metadata."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Error: TSV file has no header row.", file=sys.stderr)
            sys.exit(1)

        obj_col = None
        for col in fieldnames:
            if col.strip().lower() == "objective":
                obj_col = col
                break
        if not obj_col:
            print(f"Error: No 'Objective' column found. Columns: {fieldnames}", file=sys.stderr)
            sys.exit(1)

        for i, row in enumerate(reader):
            objective = row.get(obj_col, "").strip()
            if not objective:
                continue
            rows.append({
                "index": i,
                "objective": objective,
                "subject": row.get("Subject", row.get("subject", "")).strip(),
                "system": row.get("System", row.get("system", "")).strip(),
                "topic": row.get("Topic", row.get("topic", "")).strip(),
            })
    return rows


def build_tag(row):
    """Build an Anki tag from row metadata: Subject::System::Topic."""
    parts = [row["subject"], row["system"], row["topic"]]
    parts = [p.replace(" ", "_") for p in parts if p]
    return "::".join(parts) if parts else ""


def parse_card_lines(text, tag, custom_id):
    """Parse TSV card lines from model output. Returns list of (front, back, tag)."""
    cards = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            front, back = parts[0].strip(), parts[1].strip()
            if front and back:
                cards.append((front, back, tag))
        else:
            print(f"  Skipping malformed line from {custom_id}: {line[:80]}", file=sys.stderr)
    return cards


def chunk_list(lst, size):
    """Split list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def normalize_front(text):
    """Normalize card front text for dedup comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_cards(cards):
    """Remove cards with duplicate normalized Front text. Returns (deduped, removed_count)."""
    seen = {}
    deduped = []
    removed = 0

    for front, back, tag in cards:
        norm = normalize_front(front)
        if norm in seen:
            removed += 1
            print(f"  Duplicate removed: \"{front[:60]}\" (same as \"{seen[norm][:60]}\")", file=sys.stderr)
        else:
            seen[norm] = front
            deduped.append((front, back, tag))

    return deduped, removed


def write_output(path, cards):
    """Write cards to TSV file with header."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Front", "Back", "Tags"])
        for front, back, tag in cards:
            writer.writerow([front, back, tag])


def save_partial(path, cards):
    """Save intermediate results to partial file."""
    write_output(path + ".partial", cards)


def save_state(path, batch_ids, rows, provider_name, model):
    """Save batch IDs and row data for resume capability."""
    state = {
        "batch_ids": batch_ids,
        "rows": rows,
        "provider": provider_name,
        "model": model,
    }
    with open(path + ".state.json", "w") as f:
        json.dump(state, f)


def load_state(path):
    """Load saved state if it exists. Returns dict or None."""
    state_path = path + ".state.json"
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        return json.load(f)


def cleanup_state(path):
    """Remove state and partial files after successful completion."""
    for suffix in [".state.json", ".partial"]:
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def resolve_api_key(args, provider):
    """Resolve API key: --api-key flag > env var."""
    if args.api_key:
        return args.api_key
    key = os.environ.get(provider.env_var)
    if not key:
        print(
            f"Error: No API key found. Provide --api-key or set {provider.env_var} "
            f"(or add it to a .env file).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def main():
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from medical learning objectives.")
    parser.add_argument("input", help="Input TSV file (Google Sheets export)")
    parser.add_argument("output", help="Output TSV file for Anki import")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="claude",
                        help="API provider (default: claude)")
    parser.add_argument("--model", default=None,
                        help="Override model (default depends on provider)")
    parser.add_argument("--api-key", default=None,
                        help="API key (alternative to env var / .env file)")
    parser.add_argument("--test", action="store_true",
                        help=f"Process only the first {DEFAULT_TEST_LIMIT} objectives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats and estimated cost without calling API")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: all in one batch)")
    parser.add_argument(
        "--density", type=float, default=DEFAULT_DENSITY,
        help=(
            f"Cards per objective (default: {DEFAULT_DENSITY}). "
            "Use 0.5 to only keep the highest-yield patterns, "
            "1 for exactly one card per objective, "
            "2-3 for broader coverage."
        ),
    )
    args = parser.parse_args()

    provider = PROVIDERS[args.provider]
    model = args.model or provider.default_model

    # Read input
    print(f"Reading {args.input}...", file=sys.stderr)
    rows = read_objectives(args.input)
    print(f"Found {len(rows)} objectives.", file=sys.stderr)

    if not rows:
        print("No objectives found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.test:
        rows = rows[:DEFAULT_TEST_LIMIT]
        print(f"Test mode: using first {len(rows)} objectives.", file=sys.stderr)

    # Dry run
    if args.dry_run:
        usage = provider.estimate_cost(len(rows), args.density, model)
        batches = 1 if not args.batch_size else -(-len(rows) // args.batch_size)
        avg = max(args.density, 0.5)
        print(f"\n--- Dry Run Summary ---", file=sys.stderr)
        print(f"Provider: {args.provider}", file=sys.stderr)
        print(f"Model: {model}", file=sys.stderr)
        print(f"Objectives to process: {len(rows)}", file=sys.stderr)
        print(f"Batches: {batches}", file=sys.stderr)
        print(f"Density: {args.density} (est. ~{avg:.1f} cards/objective)", file=sys.stderr)
        print(f"Estimated cards: {int(len(rows) * avg)}", file=sys.stderr)
        if usage:
            print(f"Estimated tokens: ~{usage['total_tokens']:,} ({usage['total_input_tokens']:,} in + {usage['total_output_tokens']:,} out)", file=sys.stderr)
            print(f"Estimated cost (Batch API): ${usage['cost']:.2f}", file=sys.stderr)
        else:
            print(f"Estimated cost: (no pricing data for {model})", file=sys.stderr)
        return

    # Resolve API key
    api_key = resolve_api_key(args, provider)

    # Check for resume state
    saved = load_state(args.output)
    if saved:
        batch_ids = saved["batch_ids"]
        rows = [r if isinstance(r, dict) else r for r in saved["rows"]]
        saved_provider = saved.get("provider", args.provider)
        saved_model = saved.get("model", model)
        if saved_provider != args.provider:
            print(
                f"Warning: saved state uses provider '{saved_provider}' but you specified '{args.provider}'. "
                f"Using '{saved_provider}' to resume.",
                file=sys.stderr,
            )
            provider = PROVIDERS[saved_provider]
            model = saved_model
        print(f"Resuming from saved state with {len(batch_ids)} batch(es).", file=sys.stderr)
    else:
        # Build and submit
        system_prompt = build_system_prompt(args.density)
        client = provider.create_client(api_key)
        batch_ids = provider.submit_batch(client, rows, system_prompt, model, args.batch_size)
        if not batch_ids:
            print("No batches were submitted successfully. Exiting.", file=sys.stderr)
            sys.exit(1)
        save_state(args.output, batch_ids, rows, provider.name, model)

    # Build lookup for results
    rows_by_custom_id = {f"obj_{r['index']}": r for r in rows}
    client = provider.create_client(api_key)

    # Poll
    print("\nPolling for batch completion...", file=sys.stderr)
    provider.poll_batch(client, batch_ids)

    # Retrieve results
    print("\nRetrieving results...", file=sys.stderr)
    cards, failed_ids = provider.retrieve_results(client, batch_ids, rows_by_custom_id)
    print(f"Retrieved {len(cards)} cards from initial run.", file=sys.stderr)

    # Save partial results
    if cards:
        save_partial(args.output, cards)

    # Retry failed items (one attempt)
    if failed_ids:
        print(f"\nRetrying {len(failed_ids)} failed items...", file=sys.stderr)
        retry_rows = [rows_by_custom_id[cid] for cid in failed_ids if cid in rows_by_custom_id]
        if retry_rows:
            system_prompt = build_system_prompt(args.density)
            retry_batch_ids = provider.submit_batch(client, retry_rows, system_prompt, model, None)
            if retry_batch_ids:
                provider.poll_batch(client, retry_batch_ids)
                retry_cards, still_failed = provider.retrieve_results(client, retry_batch_ids, rows_by_custom_id)
                cards.extend(retry_cards)
                print(f"Retry recovered {len(retry_cards)} additional cards.", file=sys.stderr)
                if still_failed:
                    print(f"{len(still_failed)} items still failed after retry:", file=sys.stderr)
                    for cid in still_failed:
                        print(f"  - {cid}", file=sys.stderr)

    # Deduplicate
    print("\nChecking for duplicates...", file=sys.stderr)
    cards, dup_count = deduplicate_cards(cards)

    # Write final output
    write_output(args.output, cards)
    cleanup_state(args.output)

    # Summary
    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Provider: {args.provider} ({model})", file=sys.stderr)
    print(f"Objectives processed: {len(rows)}", file=sys.stderr)
    print(f"Cards generated: {len(cards)}", file=sys.stderr)
    print(f"Duplicates removed: {dup_count}", file=sys.stderr)
    print(f"Failed objectives: {len(failed_ids)}", file=sys.stderr)
    print(f"Output written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
