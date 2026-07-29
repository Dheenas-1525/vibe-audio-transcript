import csv
import difflib
import io
import json
import logging

log = logging.getLogger(__name__)

# The ViBe question-bank CSV's fixed 14-column structure.
TEMPLATE_COLUMNS = [
    "Segment", "Question Timestamp [mm:ss]", "S.No.", "Question", "Hint",
    "Option A", "Expln-A", "Option B", "Expln-B",
    "Option C", "Expln-C", "Option D", "Expln-D", "Correct Answer",
]

RETRY_LIMIT = 2

# Verbatim master prompt — do not edit its wording. Sent as the system
# message exactly as specified; only the JSON wire-format note below is
# appended separately as a delivery-mechanics detail, not a content rule.
MASTER_PROMPT = """You are creating a question bank (QB) for the ViBe platform from the attached video transcript. Follow every rule below exactly. Do not skip the verification step at the end.
1. Segmentation
Divide the video into logical content segments based on topic shifts (typically 5–8 segments for a 5–12 minute video). Give each segment a short, descriptive title that reflects what it actually covers.
Segments must be contiguous and cover the full video in order.
2. Question count and structure
Write exactly 5 questions per segment.
Per segment: 1 recall question (direct, low-inference restatement of a stated fact)
4 application/analysis questions (require connecting or reasoning about the content, not just restating it).
Across the whole QB, aim for roughly 50% straightforward factual statements and 50% statements with a "twist" (a plausible-sounding but incorrect claim, a swapped detail, an inverted cause-effect, etc.).
3. Question format — 100% True/False
Option A = "True", Option B = "False". Options C and D are always empty.
Correct Answer is always "A" or "B".
Across the full QB, the True/False (A/B) split should land within 45%–55% either way.
Each question is 15–30 words, written as a standalone declarative statement (a claim the learner judges true or false) — never append "True or False?" to the question text.
Never attribute the claim to the source — do not write "the video says," "the speaker states," "according to the talk," "he/she says," etc. State the claim directly as fact and let the learner judge it against what they watched.
Do not reconstruct the transcript's exact wording — paraphrase into your own sentence structure while preserving the factual content precisely.
4. Named-character scenarios
About 20% of questions (roughly 1 in 5) should frame the claim through a named individual making a decision, assumption, or statement (e.g., "Priya, a hospital administrator, tells her team that...").
Use Indian names for these (e.g., Priya, Arjun, Meera, Karthik, Anjali, Ravi, Divya, Vikram, Nisha, Suresh, Ananya, Rohan, Deepika).
The remaining ~80% of questions should be direct claims with no invented character.
Do not cluster all named-character questions in one segment — spread them across the QB.
5. Timestamp placement — the "already taught" rule
This is the most error-prone step; follow it exactly.
Transcript timestamps mark the start of a sentence/line, never its end. There is no reliable way to know exactly when a line finishes — only when the next line begins.
The "Question Timestamp" is the point on the video timeline where the question is posed to the learner. It must fall only after every piece of content the question depends on has already finished playing.
Rule: for each question, identify the last transcript line whose content the question needs. The question's timestamp must be at or after the start time of the very next transcript line after that one. This next-line start time is the safe floor — never use the same line's own start time, and never estimate an arbitrary "couple of seconds later" buffer.
A question's timestamp must stay within its own segment's time window. If the safe floor for a question's content would fall in the next segment, either move the question to that segment or rewrite it to depend only on content already covered within its current segment.
6. Content rules
No two questions in the QB may test the same underlying concept/claim in different words (check for near-duplicates before finalizing — this is different from having a recall + twist pair on the same fact, which is fine; the issue is two different questions covering the same point).
Ground every question strictly in what the transcript actually states — do not invent facts, statistics, or claims not present in the source.
For factual/statistic questions, quote numbers and specifics accurately.
7. Output format — exact CSV/spreadsheet columns, in this order
Segment
Question Timestamp [mm:ss]
S.No.
Question
Hint
Option A
Expln-A
Option B
Expln-B
Option C (always empty)
Expln-C (always empty)
Option D (always empty)
Expln-D (always empty)
Correct Answer
Hint: a short nudge toward what to recall, without giving away the answer.
Expln-A / Expln-B: explain why that option is correct or incorrect, referencing the actual content — not generic filler.
8. Mandatory verification before delivering the QB
Before finalizing, explicitly check and report:
Every question is 15–30 words.
No question uses "video says" / "speaker states" / similar attribution phrasing.
No question ends in "True or False?"
Every question's timestamp is at or after the next-line-start floor for its last dependent transcript line (show this check, don't just assert it).
Every timestamp falls within its own segment's time window.
No two questions test the same underlying concept.
Named-character questions are ~20% of the total, spread across segments.
Overall True/False (A/B) split is within 45%–55%.
Each segment has exactly 1 recall + 4 application questions.
Deliver the final QB as a spreadsheet (CSV/XLSX) with the exact 14 columns above, plus a short summary confirming each verification check passed."""

# Delivery-mechanics only — not a content rule, and not part of the prompt
# above. Tells the model to hand back the same 14-column data as JSON
# instead of literal CSV/XLSX bytes, since only structured output can be
# parsed reliably by the app.
_JSON_WIRE_FORMAT_NOTE = """

---
Wire-format note (delivery mechanics only, does not change any rule above): instead of literal CSV/XLSX bytes, return a single JSON object of exactly this shape and nothing else:
{"segments": [{"title": "string", "questions": [{"question": "string", "timestamp": "mm:ss", "hint": "string", "expln_a": "string", "expln_b": "string", "correct_answer": "A"}, ...exactly 5 per segment]}, ...], "verification_summary": "string"}
The caller fills in the literal Option A/Option B labels and the blank Option C/D/Expln-C/Expln-D columns, and assigns S.No. sequentially — omit those from your JSON."""


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


def _build_transcript_block(segments: list[dict]) -> str:
    return "\n".join(f"[{fmt_mmss(s['start'])}] {s['text'].strip()}" for s in segments)


def _validate_qb_response(parsed) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    segments = parsed.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("response missing a non-empty 'segments' list")
    for seg in segments:
        if not isinstance(seg, dict) or not isinstance(seg.get("title"), str) or not seg["title"].strip():
            raise ValueError("segment missing a 'title'")
        questions = seg.get("questions")
        if not isinstance(questions, list) or len(questions) != 5:
            got = len(questions) if isinstance(questions, list) else "no"
            raise ValueError(f"segment '{seg.get('title')}' has {got} questions, need exactly 5")
        for q in questions:
            if not isinstance(q, dict) or any(
                k not in q for k in ("question", "timestamp", "hint", "expln_a", "expln_b", "correct_answer")
            ):
                raise ValueError("question item missing required fields")
            if q["correct_answer"] not in ("A", "B"):
                raise ValueError(f"correct_answer must be 'A' or 'B', got {q['correct_answer']!r}")
            if _parse_mmss(q["timestamp"]) is None:
                raise ValueError(f"unparseable timestamp {q['timestamp']!r}")
    return {"segments": segments, "verification_summary": parsed.get("verification_summary", "")}


# Independent, code-side re-check of the prompt's own rules (Section 8),
# since the model's self-reported verification_summary is not otherwise
# fact-checked. Soft/informational only — does not trigger a retry, since
# retrying the entire multi-segment call over one flagged item is wasteful;
# these are surfaced in the summary for a human to review instead.
_BANNED_ATTRIBUTION_PHRASES = [
    "the video", "the speaker", "the lecture", "the transcript",
    "the source material", "according to the talk", "he says", "she says",
]
_EXAMPLE_NAMES = [
    "Priya", "Arjun", "Meera", "Karthik", "Anjali", "Ravi", "Divya",
    "Vikram", "Nisha", "Suresh", "Ananya", "Rohan", "Deepika",
]
_DUPLICATE_SIMILARITY_THRESHOLD = 0.75


def _code_verify(data: dict, rows: list[dict]) -> list[str]:
    warnings = []
    total = len(rows)
    if total == 0:
        return warnings

    for r in rows:
        wc = len(r["Question"].split())
        if not (15 <= wc <= 30):
            warnings.append(f"S.No. {r['S.No.']}: question is {wc} words (outside 15-30).")
        qlow = r["Question"].lower()
        if any(p in qlow for p in _BANNED_ATTRIBUTION_PHRASES):
            warnings.append(f"S.No. {r['S.No.']}: question attributes the claim to the source.")
        if qlow.strip().endswith("true or false?"):
            warnings.append(f"S.No. {r['S.No.']}: question ends with \"True or False?\".")

    ab_counts = {"A": 0, "B": 0}
    for r in rows:
        if r.get("Correct Answer") in ab_counts:
            ab_counts[r["Correct Answer"]] += 1
    a_ratio = ab_counts["A"] / total
    if not (0.45 <= a_ratio <= 0.55):
        warnings.append(f"A/B split is {a_ratio:.0%} True, outside the 45-55% target.")

    named_flags = [any(n in r["Question"] for n in _EXAMPLE_NAMES) for r in rows]
    named_ratio = sum(named_flags) / total
    if not (0.10 <= named_ratio <= 0.30):
        warnings.append(f"Named-character questions are {named_ratio:.0%} of the total (expected roughly 20%).")
    by_segment: dict[str, int] = {}
    for r, flag in zip(rows, named_flags):
        by_segment[r["Segment"]] = by_segment.get(r["Segment"], 0) + (1 if flag else 0)
    for seg, count in by_segment.items():
        if count > 1:
            warnings.append(f"Segment '{seg}' has {count} named-character questions clustered in one segment.")

    questions = [r["Question"] for r in rows]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            ratio = difflib.SequenceMatcher(None, questions[i].lower(), questions[j].lower()).ratio()
            if ratio > _DUPLICATE_SIMILARITY_THRESHOLD:
                warnings.append(
                    f"S.No. {rows[i]['S.No.']} and S.No. {rows[j]['S.No.']} may test the same "
                    f"underlying concept (similarity {ratio:.0%})."
                )

    prev_max = -1.0
    for seg in data["segments"]:
        times = [t for t in (_parse_mmss(q["timestamp"]) for q in seg["questions"]) if t is not None]
        if not times:
            continue
        if min(times) < prev_max:
            warnings.append(f"Segment '{seg['title']}' has a question timestamped before the previous segment ended.")
        prev_max = max(prev_max, max(times))

    return warnings


def generate_question_bank(
    segments: list[dict],
    template_columns: list[str],
    client,
    model: str,
    progress_cb=None,
):
    """Runs the ViBe master prompt over the full transcript in a single call
    and returns (rows, summary)."""
    if not segments:
        return [], {"warnings": ["No transcript segments to generate questions from."]}

    if progress_cb:
        progress_cb(0, 1)

    transcript_block = _build_transcript_block(segments)
    messages = [
        {"role": "system", "content": MASTER_PROMPT + _JSON_WIRE_FORMAT_NOTE},
        {"role": "user", "content": transcript_block},
    ]

    data, last_err = None, None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            parsed = json.loads(resp.choices[0].message.content)
            data = _validate_qb_response(parsed)
            break
        except Exception as e:
            last_err = e
            log.warning(f"  question bank generation attempt {attempt + 1} failed: {e}")

    if data is None:
        raise RuntimeError(f"question bank generation failed after retries: {last_err}")

    rows = []
    sno = 1
    for seg in data["segments"]:
        for q in seg["questions"]:
            values = {
                "Segment": seg["title"],
                "Question Timestamp [mm:ss]": q["timestamp"],
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

    if progress_cb:
        progress_cb(1, 1)

    ab_counts = {"A": 0, "B": 0}
    for r in rows:
        ans = r.get("Correct Answer")
        if ans in ab_counts:
            ab_counts[ans] += 1

    code_warnings = _code_verify(data, rows)

    summary = {
        "video_length": fmt_mmss(segments[-1]["end"]),
        "segment_count": len(data["segments"]),
        "question_count": len(rows),
        "ab_split": ab_counts,
        "verification_summary": data["verification_summary"],
        "code_warnings": code_warnings,
    }
    return rows, summary


def rows_to_csv_str(rows: list[dict], template_columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(template_columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in template_columns])
    return buf.getvalue()
