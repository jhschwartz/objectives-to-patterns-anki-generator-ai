#!/usr/bin/env python3
"""Generate Anki flashcards from medical learning objectives using the Claude Batches API."""

import argparse
import csv
import json
import os
import re
import sys
import time

import anthropic

MODEL = "claude-sonnet-4-20250514"
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
- Do NOT include the header row "Front\tBack" in your response
- If the objective doesn't lend itself to a good clinical pattern card, return an empty response"""


def get_density_instruction(density):
    """Return the prompt instruction fragment for the given card density.

    Treats density as a continuous value:
      <0.5  — very selective, skip low-yield objectives entirely
      0.5-1 — aim for ~1 card, skip some low-yield objectives
      1     — exactly 1 card per objective
      1-2   — 1-2 cards
      2+    — up to N cards, broader coverage
    """
    if density < 1.0:
        # Sub-1: tell Claude to be selective and sometimes return nothing
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

        # Find the Objective column (case-insensitive)
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


def build_batch_requests(rows, density=DEFAULT_DENSITY):
    """Build list of batch request dicts for the Batches API."""
    system_prompt = build_system_prompt(density)
    requests = []
    for row in rows:
        requests.append({
            "custom_id": f"obj_{row['index']}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": row["objective"],
                    }
                ],
            },
        })
    return requests


def chunk_list(lst, size):
    """Split list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def submit_batches(client, all_requests, batch_size):
    """Submit batch request(s) and return list of batch IDs."""
    batch_ids = []
    if batch_size and batch_size < len(all_requests):
        chunks = list(chunk_list(all_requests, batch_size))
    else:
        chunks = [all_requests]

    for i, chunk in enumerate(chunks):
        print(f"Submitting batch {i + 1}/{len(chunks)} ({len(chunk)} requests)...", file=sys.stderr)
        try:
            batch = client.messages.batches.create(requests=chunk)
            batch_ids.append(batch.id)
            print(f"  Batch ID: {batch.id}", file=sys.stderr)
        except anthropic.APIError as e:
            print(f"  Error submitting batch {i + 1}: {e}", file=sys.stderr)
            # Retry once with backoff
            time.sleep(10)
            try:
                batch = client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"  Batch ID (retry): {batch.id}", file=sys.stderr)
            except anthropic.APIError as e2:
                print(f"  Failed after retry: {e2}", file=sys.stderr)
    return batch_ids


def poll_batches(client, batch_ids):
    """Poll until all batches reach 'ended' status. Returns list of batch objects."""
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


def retrieve_results(client, batch_ids, rows_by_custom_id):
    """Retrieve results from completed batches. Returns (cards, failed_ids)."""
    cards = []
    failed_ids = []

    for batch_id in batch_ids:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            row = rows_by_custom_id.get(custom_id)
            tag = build_tag(row) if row else ""

            if result.result.type == "succeeded":
                message = result.result.message
                text = ""
                for block in message.content:
                    if block.type == "text":
                        text += block.text

                for line in text.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Split on first tab only
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        front, back = parts[0].strip(), parts[1].strip()
                        if front and back:
                            cards.append((front, back, tag))
                    else:
                        print(f"  Skipping malformed line from {custom_id}: {line[:80]}", file=sys.stderr)
            else:
                failed_ids.append(custom_id)
                error_type = result.result.type
                print(f"  {custom_id} failed: {error_type}", file=sys.stderr)

    return cards, failed_ids


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
    partial_path = path + ".partial"
    write_output(partial_path, cards)


def save_state(path, batch_ids, rows):
    """Save batch IDs and row data for resume capability."""
    state_path = path + ".state.json"
    state = {
        "batch_ids": batch_ids,
        "rows": rows,
    }
    with open(state_path, "w") as f:
        json.dump(state, f)


def load_state(path):
    """Load saved state if it exists. Returns (batch_ids, rows) or None."""
    state_path = path + ".state.json"
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        state = json.load(f)
    return state["batch_ids"], state["rows"]


def cleanup_state(path):
    """Remove state and partial files after successful completion."""
    for suffix in [".state.json", ".partial"]:
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def estimate_usage(num_objectives, density=DEFAULT_DENSITY):
    """Estimate token usage and API cost for batch processing.

    Returns dict with token counts and dollar cost.
    """
    # System prompt: ~500 tokens (cached after first request in batch)
    # Each objective: ~100 input tokens + ~200 output tokens (scales with density)
    system_tokens = 500
    input_per_request = 100
    output_per_request = int(200 * max(density, 0.5) / DEFAULT_DENSITY)

    # First request pays full input for system prompt, rest use cache
    full_input_tokens = system_tokens + input_per_request
    cached_input_tokens = input_per_request  # system prompt cached
    total_input_tokens = full_input_tokens + (num_objectives - 1) * (cached_input_tokens + system_tokens)
    total_output_tokens = num_objectives * output_per_request

    # Batch API pricing (Sonnet): $3/M input, $15/M output, 50% discount for batches
    # Cached input: $0.30/M (90% discount on input)
    input_cost = (full_input_tokens * 1.5 / 1_000_000) + (
        (num_objectives - 1) * cached_input_tokens * 1.5 / 1_000_000
    )
    cached_cost = (num_objectives - 1) * system_tokens * 0.15 / 1_000_000
    output_cost = total_output_tokens * 7.5 / 1_000_000

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cost": input_cost + cached_cost + output_cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from medical learning objectives.")
    parser.add_argument("input", help="Input TSV file (Google Sheets export)")
    parser.add_argument("output", help="Output TSV file for Anki import")
    parser.add_argument("--test", action="store_true", help=f"Process only the first {DEFAULT_TEST_LIMIT} objectives")
    parser.add_argument("--dry-run", action="store_true", help="Show stats and estimated cost without calling API")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: all in one batch)")
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
        usage = estimate_usage(len(rows), density=args.density)
        batches = 1 if not args.batch_size else -(-len(rows) // args.batch_size)  # ceil division
        avg = max(args.density, 0.5)
        print(f"\n--- Dry Run Summary ---", file=sys.stderr)
        print(f"Objectives to process: {len(rows)}", file=sys.stderr)
        print(f"Batches: {batches}", file=sys.stderr)
        print(f"Density: {args.density} (est. ~{avg:.1f} cards/objective)", file=sys.stderr)
        print(f"Estimated cards: {int(len(rows) * avg)}", file=sys.stderr)
        print(f"Estimated tokens: ~{usage['total_tokens']:,} ({usage['total_input_tokens']:,} in + {usage['total_output_tokens']:,} out)", file=sys.stderr)
        print(f"Estimated cost (Batch API): ${usage['cost']:.2f}", file=sys.stderr)
        print(f"Model: {MODEL}", file=sys.stderr)
        return

    # Check for resume state
    saved = load_state(args.output)
    if saved:
        batch_ids, saved_rows = saved
        rows = [r if isinstance(r, dict) else r for r in saved_rows]
        print(f"Resuming from saved state with {len(batch_ids)} batch(es).", file=sys.stderr)
    else:
        # Build and submit
        rows_by_custom_id = {f"obj_{r['index']}": r for r in rows}
        all_requests = build_batch_requests(rows, density=args.density)
        client = anthropic.Anthropic()

        batch_ids = submit_batches(client, all_requests, args.batch_size)
        if not batch_ids:
            print("No batches were submitted successfully. Exiting.", file=sys.stderr)
            sys.exit(1)

        save_state(args.output, batch_ids, rows)

    # Build lookup for results
    rows_by_custom_id = {f"obj_{r['index']}": r for r in rows}
    client = anthropic.Anthropic()

    # Poll
    print("\nPolling for batch completion...", file=sys.stderr)
    poll_batches(client, batch_ids)

    # Retrieve results
    print("\nRetrieving results...", file=sys.stderr)
    cards, failed_ids = retrieve_results(client, batch_ids, rows_by_custom_id)
    print(f"Retrieved {len(cards)} cards from initial run.", file=sys.stderr)

    # Save partial results
    if cards:
        save_partial(args.output, cards)

    # Retry failed items (one attempt)
    if failed_ids:
        print(f"\nRetrying {len(failed_ids)} failed items...", file=sys.stderr)
        retry_rows = [rows_by_custom_id[cid] for cid in failed_ids if cid in rows_by_custom_id]
        if retry_rows:
            retry_requests = build_batch_requests(retry_rows, density=args.density)
            retry_batch_ids = submit_batches(client, retry_requests, None)
            if retry_batch_ids:
                poll_batches(client, retry_batch_ids)
                retry_cards, still_failed = retrieve_results(client, retry_batch_ids, rows_by_custom_id)
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
    print(f"Objectives processed: {len(rows)}", file=sys.stderr)
    print(f"Cards generated: {len(cards)}", file=sys.stderr)
    print(f"Duplicates removed: {dup_count}", file=sys.stderr)
    print(f"Failed objectives: {len(failed_ids)}", file=sys.stderr)
    print(f"Output written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
