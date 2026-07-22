import csv
import io
import json
import logging
import re

log = logging.getLogger(__name__)

# Safety net for the curly-quote rule: converts a straight-quoted 'phrase'
# into curly ‘phrase’. Only matches a genuine open/close PAIR (3-200 chars,
# no embedded newline or further apostrophe), so a lone contraction/possessive
# apostrophe (don't, Priya's) never matches — it has no closing partner.
# Primary defense is still the prompt instructing the model to emit curly
# quotes natively; this only catches the cases where it doesn't.
_QUOTE_PAIR = re.compile(r"(?<!\w)'([^'\n]{3,200}?)'(?!\w)")


def _wrap_curly_quotes(text: str) -> str:
    return _QUOTE_PAIR.sub("‘\\1’", text)

RETRY_LIMIT = 2

TARGET_SEGMENT_SECONDS = 150   # ~2:30 aim
MAX_SEGMENT_SECONDS = 180      # 3:00 soft target ceiling
HARD_CEILING_SECONDS = 210     # 3:30 hard ceiling — enforced in code regardless of LLM output
MIN_SEGMENT_SECONDS = 90       # 1:30 floor — code merges anything shorter, regardless of LLM output

ROLE_NAMES = [
    "Priya", "Arjun", "Meera", "Karthik", "Anjali",
    "Ravi", "Divya", "Vikram", "Nisha", "Suresh",
]

REQUIRED_ITEM_FIELDS = ["question", "hint", "expln_a", "expln_b", "correct_answer"]

# Style anchor: real rows from a hand-made question bank, used to ground tone
# and register. The written rules in _segment_prompt() are authoritative for
# exact mechanics (word counts, quote formatting) where they differ from this
# older example.
STYLE_EXAMPLE_ROWS = [
    {
        "question": "A junior chemist describes soap as the sodium or potassium salt of "
        "long-chain fatty acids produced by base hydrolysis of fats or oils.",
        "hint": "What is the chemical identity of soap?",
        "expln_a": "Correct. Soap is precisely defined as the sodium or potassium salt of "
        "long-chain fatty acids. The long hydrocarbon chain is hydrophobic, while the "
        "carboxylate end is hydrophilic — this dual nature is what gives soap its cleaning "
        "power. Base hydrolysis of fats or oils, known as saponification, is the reaction "
        "that produces these salts.",
        "expln_b": "Incorrect. This is the accurate definition of soap. Soaps are "
        "‘sodium or potassium salts of fatty acids’ where fatty acids refer to "
        "long-chain carboxylic acids. The molecule has a hydrophobic hydrocarbon tail and a "
        "hydrophilic carboxylate head — that dual structure is foundational to everything "
        "else about how soap works.",
        "correct_answer": "A",
    },
    {
        "question": "Karthik, a process engineer, explains that saponification is the acid "
        "hydrolysis of triglycerides using hydrochloric acid to produce soap and glycerol.",
        "hint": "Is acid hydrolysis the correct route for soap production?",
        "expln_a": "Incorrect. Saponification is base hydrolysis, not acid hydrolysis. A "
        "fat or oil reacts with a base — typically sodium hydroxide — to produce soap and "
        "glycerol. Acid hydrolysis of triglycerides yields free fatty acids instead, not "
        "soap salts. Karthik has confused two different reaction pathways.",
        "expln_b": "Correct. Karthik has it backwards. Saponification uses a base such as "
        "sodium hydroxide, not an acid — ‘base hydrolysis is also called "
        "saponification,’ producing soap and glycerol. Acid hydrolysis splits the same "
        "ester bonds but gives free fatty acids, not the salts that constitute soap.",
        "correct_answer": "B",
    },
]


def fmt_mmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_mmss(value):
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        parts = [int(p) for p in value.strip().split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _is_terminal_punctuation(text: str) -> bool:
    """True if text ends on a real sentence break (. ! ?), ignoring any
    trailing quote/bracket/space around the punctuation."""
    stripped = text.rstrip().rstrip('"\'”’)]').rstrip()
    return stripped.endswith((".", "!", "?"))


def _snap_to_nearest(target: float, candidates: list[float], punctuated: set = None, tolerance: float = 20.0) -> float:
    """Snap to the nearest candidate timestamp, preferring one that lands on
    terminal punctuation (a real sentence break) if one exists within
    `tolerance` seconds of the target."""
    if punctuated:
        near_punct = [c for c in candidates if c in punctuated and abs(c - target) <= tolerance]
        if near_punct:
            return min(near_punct, key=lambda c: abs(c - target))
    return min(candidates, key=lambda c: abs(c - target))


def _pick_split_point(window: list[float], punctuated: set) -> float:
    """Among candidate split points, prefer the latest one that lands on
    terminal punctuation; fall back to the latest overall if none qualify."""
    punct_window = [t for t in window if t in punctuated]
    return max(punct_window) if punct_window else max(window)


def _enforce_hard_ceiling(boundaries: list[float], end_times: list[float], punctuated: set):
    """Guarantee no segment exceeds HARD_CEILING_SECONDS, splitting at real
    sentence-end timestamps (preferring ones on terminal punctuation) if a
    proposed beat came in too long. Returns (final_boundaries,
    forced_split_values) so callers can report which splits were code-forced
    rather than proposed by the model."""
    result = []
    forced = set()
    prev = 0.0
    for b in boundaries:
        cursor = prev
        while b - cursor > HARD_CEILING_SECONDS:
            window = [t for t in end_times if cursor < t <= cursor + MAX_SEGMENT_SECONDS]
            if not window:
                window = [t for t in end_times if cursor < t <= cursor + HARD_CEILING_SECONDS]
            if not window:
                break  # no sentence break available to split on; accept the long segment
            split_at = _pick_split_point(window, punctuated)
            result.append(split_at)
            forced.add(split_at)
            cursor = split_at
        result.append(b)
        prev = b
    final = []
    for b in result:
        if not final or b > final[-1]:
            final.append(b)
    return final, forced


def _merge_short_segments(boundaries: list[float]) -> list[float]:
    """Collapse consecutive boundaries into windows of at least
    MIN_SEGMENT_SECONDS, so a model that (mis)proposes a boundary at every
    sentence doesn't produce dozens of tiny segments. Always keeps the final
    boundary intact even if the trailing remainder is short."""
    if not boundaries:
        return boundaries
    merged = []
    window_start = 0.0
    for b in boundaries:
        if b - window_start >= MIN_SEGMENT_SECONDS:
            merged.append(b)
            window_start = b
    if not merged or merged[-1] != boundaries[-1]:
        merged.append(boundaries[-1])
    return merged


def plan_segment_boundaries(segments: list[dict], client, model: str):
    """Ask the model to propose topic/teachable-beat segment boundaries, snap
    each to a real transcript sentence-end (preferring ones on terminal
    punctuation), then merge anything too short and force-split anything
    still over the hard ceiling — so the numeric rules hold regardless of
    what the model actually returns. Returns (boundaries, forced_split_values).
    """
    if not segments:
        return [], set()

    end_times = [s["end"] for s in segments]
    punctuated = {s["end"] for s in segments if _is_terminal_punctuation(s["text"])}
    total_duration = end_times[-1]

    # A video shorter than the target max doesn't need splitting at all —
    # skip the LLM call entirely rather than risk it over-segmenting.
    if total_duration <= MAX_SEGMENT_SECONDS:
        return [total_duration], set()

    transcript_block = "\n".join(f"[{fmt_mmss(s['end'])}] {s['text']}" for s in segments)

    system = (
        "You divide a lecture transcript into topic-based teaching segments "
        "(\"teachable beats\") for an in-video pause-and-ask quiz tool.\n"
        "- Aim for one segment per ~2-3 minutes of video (target 2:30). A "
        f"{fmt_mmss(total_duration)} video should produce roughly "
        f"{max(1, round(total_duration / TARGET_SEGMENT_SECONDS))} segments — "
        "boundaries must be FAR FEWER than the number of transcript lines shown. "
        "You are choosing a small number of major topic breaks, NOT listing every "
        "timestamp in the transcript.\n"
        "- Target maximum segment length is 3:00; never exceed 3:30. Never go "
        f"below {fmt_mmss(MIN_SEGMENT_SECONDS)} either, except for the final segment.\n"
        "- Each boundary MUST be one of the exact [mm:ss] timestamps shown in the "
        "transcript (these mark real sentence breaks) — never invent a timestamp "
        "that isn't listed there.\n"
        "- The final boundary must equal the transcript's last timestamp.\n"
        "- Prefer boundaries at natural topic changes, not mid-explanation.\n\n"
        "Respond with a JSON object of exactly this shape and nothing else: "
        '{"boundaries": ["mm:ss", "mm:ss", ...]}'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Transcript:\n{transcript_block}"},
    ]

    boundaries_sec = None
    last_err = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            data = json.loads(resp.choices[0].message.content)
            raw = data.get("boundaries") if isinstance(data, dict) else None
            if not isinstance(raw, list) or not raw:
                raise ValueError("model response missing a 'boundaries' list")
            parsed = [_parse_mmss(b) for b in raw]
            boundaries_sec = [b for b in parsed if b is not None]
            if not boundaries_sec:
                raise ValueError("no parseable boundary timestamps")
            break
        except Exception as e:
            last_err = e
            log.warning(f"  segment planning attempt {attempt + 1} failed: {e}")

    if boundaries_sec is None:
        log.warning(f"  segment planning failed after retries ({last_err}); using fixed windows")
        boundaries_sec = list(
            range(TARGET_SEGMENT_SECONDS, int(total_duration) + 1, TARGET_SEGMENT_SECONDS)
        )

    snapped = sorted(set(_snap_to_nearest(b, end_times, punctuated) for b in boundaries_sec))
    if not snapped or snapped[-1] != total_duration:
        snapped.append(total_duration)

    merged = _merge_short_segments(snapped)
    return _enforce_hard_ceiling(merged, end_times, punctuated)


BANNED_META_PHRASES = ["the lecture", "the video", "the speaker", "in the lecture"]

# Heuristic: phrases near the end of a question that signal the answer
# instead of letting the reader judge for themselves.
CLOSING_SIGNAL_PHRASES = [
    "matches what was taught", "matches the lecture", "as taught", "as described",
    "as explained", "correctly reflects", "accurately represents", "in line with",
    "consistent with what was", "which is correct", "which is incorrect",
]


def _ends_with_signal_phrase(question: str) -> bool:
    tail = " ".join(question.lower().split()[-10:])
    return any(p in tail for p in CLOSING_SIGNAL_PHRASES)


def build_summary(
    rows: list[dict],
    boundaries: list[float],
    forced_splits: set,
    punctuated: set,
    video_duration: float,
    questions_per_segment: int,
) -> dict:
    """Runs the spec's self-check list against the final output and returns a
    report — segment/question counts, A/B split, max segment length, a
    sample row, and any compliance warnings found. Some of these rules are
    also enforced as retry triggers in _validate_questions (count, prefix,
    banned phrases); they're re-checked here too as a final, honest report
    of what actually shipped, including any item that survived retries."""
    question_count = len(rows)
    ab_counts = {"A": 0, "B": 0}
    for r in rows:
        ans = r.get("Correct Answer")
        if ans in ab_counts:
            ab_counts[ans] += 1

    spans = []
    prev = 0.0
    for b in boundaries:
        spans.append(b - prev)
        prev = b
    max_span = max(spans) if spans else 0.0

    warnings = []
    if boundaries and abs(boundaries[-1] - video_duration) > 1e-3:
        warnings.append("Last anchor does not equal the video's final timestamp.")
    if max_span > HARD_CEILING_SECONDS + 1e-6:
        warnings.append(f"A segment exceeded the 3:30 hard ceiling ({fmt_mmss(max_span)}).")
    elif max_span > MAX_SEGMENT_SECONDS + 1e-6:
        warnings.append(f"A segment exceeded the 3:00 target (within the 3:30 ceiling): {fmt_mmss(max_span)}.")
    for b in boundaries:
        if b not in punctuated:
            warnings.append(f"Anchor at {fmt_mmss(b)} does not land on terminal punctuation.")
    if question_count:
        a_ratio = ab_counts["A"] / question_count
        if not (0.45 <= a_ratio <= 0.55):
            warnings.append(f"A/B balance is {a_ratio:.0%} True, outside the 45-55% target.")

    by_segment: dict[str, list[dict]] = {}
    for r in rows:
        by_segment.setdefault(r.get("Segment"), []).append(r)
    for seg, seg_rows in by_segment.items():
        if len(seg_rows) != questions_per_segment:
            warnings.append(f"Segment {seg} has {len(seg_rows)} questions, expected {questions_per_segment}.")
        timestamps = {r.get("Question Timestamp [mm:ss]") for r in seg_rows}
        if len(timestamps) > 1:
            warnings.append(f"Segment {seg} questions don't all share one timestamp: {timestamps}.")

    for r in rows:
        q = r.get("Question", "")
        q_lower = q.lower()
        if any(p in q_lower for p in BANNED_META_PHRASES):
            warnings.append(f"S.No. {r.get('S.No.')}: question contains a meta-reference (\"the lecture\"/\"the video\").")
        if _ends_with_signal_phrase(q):
            warnings.append(f"S.No. {r.get('S.No.')}: question ends with an answer-signaling phrase.")
        for col in ("Expln-A", "Expln-B"):
            text = r.get(col, "")
            wc = len(text.split())
            if not (40 <= wc <= 90):
                warnings.append(f"S.No. {r.get('S.No.')}: {col} is {wc} words (outside 40-90).")
            if not (text.startswith("Correct.") or text.startswith("Incorrect.")):
                warnings.append(f"S.No. {r.get('S.No.')}: {col} doesn't start with \"Correct.\"/\"Incorrect.\".")
        if r.get("Correct Answer") not in ("A", "B"):
            warnings.append(f"S.No. {r.get('S.No.')}: Correct Answer is not A or B.")

    return {
        "video_length": fmt_mmss(video_duration),
        "segment_count": len(boundaries),
        "question_count": question_count,
        "ab_split": ab_counts,
        "max_segment_length": fmt_mmss(max_span),
        "forced_split_count": len(forced_splits),
        "sample_row": rows[0] if rows else None,
        "warnings": warnings,
    }


def chunk_segments_by_boundaries(segments: list[dict], boundaries: list[float]):
    """Group transcript segments into finalized (end_time, text) chunks matching
    the planned boundaries."""
    if not boundaries:
        return [(segments[-1]["end"], " ".join(s["text"] for s in segments))] if segments else []
    chunks = []
    b_idx = 0
    current = []
    for seg in segments:
        current.append(seg)
        if b_idx < len(boundaries) and seg["end"] >= boundaries[b_idx] - 1e-6:
            chunks.append((boundaries[b_idx], " ".join(s["text"] for s in current)))
            current = []
            b_idx += 1
    if current:
        chunks.append((boundaries[-1], " ".join(s["text"] for s in current)))
    return chunks


def _segment_prompt(n_questions: int, true_ratio_so_far: float, named_position: int) -> str:
    examples = "\n\n".join(
        f"Example:\nQuestion: {r['question']}\nHint: {r['hint']}\n"
        f"Option A: True | Expln-A: {r['expln_a']}\n"
        f"Option B: False | Expln-B: {r['expln_b']}\n"
        f"Correct Answer: {r['correct_answer']}"
        for r in STYLE_EXAMPLE_ROWS
    )
    names = ", ".join(ROLE_NAMES)

    if true_ratio_so_far > 0.55:
        balance_note = (
            f"So far {true_ratio_so_far:.0%} of correct answers across this file have been "
            "True — lean toward writing more False (misapplied) scenarios in this batch to "
            "rebalance toward 50/50."
        )
    elif true_ratio_so_far < 0.45:
        balance_note = (
            f"So far only {true_ratio_so_far:.0%} of correct answers across this file have "
            "been True — lean toward writing more True (correctly applied) scenarios in this "
            "batch to rebalance toward 50/50."
        )
    else:
        balance_note = "Keep a natural, roughly even True/False split in this batch."

    return f"""You write an in-class "pause-and-ask" question bank for a video learning platform, from one topic segment of a lecture transcript.

STYLE ANCHOR (tone/register reference — the rules below take precedence wherever they differ, e.g. exact word counts and quote formatting):
{examples}

RULES FOR THIS BATCH ({n_questions} questions):
- 100% True/False format. Option A is always "True", Option B is always "False".
- The 1st question is a straightforward recall check; the remaining {n_questions - 1} require application or analysis of the idea, not just memory.
- Roughly half the questions should be straightforwardly factual, and half should have a small twist that surfaces a common misconception.
- Every question is SCENARIO-BASED: a fictional decision-maker (founder, CEO, developer, team lead, junior engineer, senior engineer) applying the segment's idea correctly or misapplying it. The reader judges which.
- NAMING IS RARE: ONLY question #{named_position} may use a name (prefer: {names}). Every other question in this batch MUST use a plain unnamed role only ("a developer", "a team lead") — do not name any of them.
- Compact: 15-30 words median per question, 38-word hard ceiling for prose-only questions. For code-related content, a fenced markdown code block (3-6 lines) is allowed within the question.
- The scenario must stand alone — never write "the lecture," "the speaker," "the video," "in the lecture," etc.
- Do NOT end the question with a phrase that signals the answer (e.g. "...which matches what was taught") — present the scenario and let the reader judge.
- {balance_note}

HINT: a short, pointed question (5-15 words) that signals what to check — never gives away the answer.

EXPLANATIONS (expln_a and expln_b): BOTH are full teaching moments, regardless of which option is correct:
- HARD REQUIREMENT: each of expln_a and expln_b must be between 40 and 90 words — count before finalizing. Under 40 words is not acceptable; pad with genuine teaching content (why it matters, what the correct concept is), not filler.
- Start with exactly "Correct." or "Incorrect." (with the period).
- Re-teach the underlying concept, not just judge the scenario. Avoid saying "the lecture" here too — quote the transcript directly instead of narrating that it was said.
- Include one short direct quote from the transcript (5-25 words), wrapped in curly typographic quotes ‘like this’ — NOT straight quotes. Keep ordinary apostrophes in contractions/possessives straight (don't, Priya's).
- End with a sharp one-line summary where it fits naturally.
- Tone: direct, slightly punchy, no hedging.

Respond with a JSON object of exactly this shape and nothing else:
{{"questions": [{{"question": "...", "hint": "...", "expln_a": "...", "expln_b": "...", "correct_answer": "A"}}, ...]}}"""


def _validate_questions(items, n_questions):
    """Raises (triggering a retry in the caller) for violations the model is
    reliably able to fix on a retry: wrong item count, missing an
    explanation prefix, or a banned meta-reference in the question. Word
    count and closing-signal-phrase are intentionally NOT retry triggers —
    those are softer/less reliably fixable and risk exhausting retries and
    failing the whole segment; they're reported instead in build_summary()."""
    valid = [
        item for item in items
        if isinstance(item, dict)
        and all(field in item for field in REQUIRED_ITEM_FIELDS)
        and item["correct_answer"] in ("A", "B")
    ]
    if len(valid) < n_questions:
        raise ValueError(f"got {len(valid)} valid items, need {n_questions}")

    valid = valid[:n_questions]
    for item in valid:
        for field in ("expln_a", "expln_b"):
            if not (item[field].startswith("Correct.") or item[field].startswith("Incorrect.")):
                raise ValueError(f"{field} doesn't start with 'Correct.'/'Incorrect.'")
        if any(p in item["question"].lower() for p in BANNED_META_PHRASES):
            raise ValueError("question contains a banned meta-reference")

    for item in valid:
        item["expln_a"] = _wrap_curly_quotes(item["expln_a"])
        item["expln_b"] = _wrap_curly_quotes(item["expln_b"])
    return valid


def _soft_violation_count(items) -> int:
    """Counts violations of rules that are real but not worth failing the
    whole segment over (word count, closing signal phrase) — used to pick
    the best attempt across retries rather than accepting the first one
    that merely passes the hard checks in _validate_questions."""
    count = 0
    for item in items:
        for field in ("expln_a", "expln_b"):
            wc = len(item[field].split())
            if not (40 <= wc <= 90):
                count += 1
        if _ends_with_signal_phrase(item["question"]):
            count += 1
    return count


def generate_questions_for_segment(
    segment_text: str,
    n_questions: int,
    true_ratio_so_far: float,
    segment_index: int,
    client,
    model: str,
):
    named_position = (segment_index % n_questions) + 1
    messages = [
        {"role": "system", "content": _segment_prompt(n_questions, true_ratio_so_far, named_position)},
        {"role": "user", "content": f"Transcript segment:\n{segment_text}"},
    ]
    best, best_score, last_err = None, None, None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            data = json.loads(resp.choices[0].message.content)
            items = data.get("questions") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError("model response missing a 'questions' array")
            valid = _validate_questions(items, n_questions)  # raises on hard violations (count/prefix/banned phrase)
        except Exception as e:
            last_err = e
            log.warning(f"  segment generation attempt {attempt + 1} failed: {e}")
            continue

        score = _soft_violation_count(valid)
        if best is None or score < best_score:
            best, best_score = valid, score
        if score == 0:
            return valid
        log.warning(f"  segment generation attempt {attempt + 1}: {score} soft violation(s) (word count/closing phrase)")

    if best is not None:
        log.warning(f"  segment generation: accepting best-effort batch with {best_score} soft violation(s) after retries")
        return best
    raise RuntimeError(f"question generation failed after retries: {last_err}")


def generate_question_bank(
    segments: list[dict],
    template_columns: list[str],
    client,
    model: str,
    questions_per_segment: int = 5,
    progress_cb=None,
):
    """Returns (rows, summary) — summary is the self-check report described
    in build_summary()."""
    if progress_cb:
        progress_cb(0, 0)  # signal "planning segments" phase before per-chunk progress starts

    boundaries, forced_splits = plan_segment_boundaries(segments, client, model)
    chunks = chunk_segments_by_boundaries(segments, boundaries)

    rows = []
    sno = 1
    true_count = 0
    total_count = 0
    for idx, (end_ts, chunk_text) in enumerate(chunks, start=1):
        true_ratio = (true_count / total_count) if total_count else 0.5
        questions = generate_questions_for_segment(
            chunk_text, questions_per_segment, true_ratio, idx, client, model
        )
        for q in questions:
            values = {
                "Segment": str(idx),
                "Question Timestamp [mm:ss]": fmt_mmss(end_ts),
                "S.No.": str(sno),
                "Question": q["question"],
                "Hint": q["hint"],
                "Option A": "True",
                "Expln-A": q["expln_a"],
                "Option B": "False",
                "Expln-B": q["expln_b"],
                "Option C": "",
                "Expln-C": "",
                "Option D": "",
                "Expln-D": "",
                "Correct Answer": q["correct_answer"],
            }
            rows.append({col: values.get(col, "") for col in template_columns})
            sno += 1
            total_count += 1
            if q["correct_answer"] == "A":
                true_count += 1
        if progress_cb:
            progress_cb(idx, len(chunks))

    video_duration = segments[-1]["end"] if segments else 0.0
    punctuated = {s["end"] for s in segments if _is_terminal_punctuation(s["text"])}
    summary = build_summary(rows, boundaries, forced_splits, punctuated, video_duration, questions_per_segment)
    return rows, summary


def rows_to_csv_str(rows: list[dict], template_columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(template_columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in template_columns])
    return buf.getvalue()
