import csv
import io
import json
import logging

log = logging.getLogger(__name__)

RETRY_LIMIT = 2

# Style anchor lifted from a real, hand-made question bank so the model
# reliably reproduces the persona / true-false / dual-explanation format.
STYLE_EXAMPLE_ROWS = [
    {
        "question": "A junior chemist describes soap as the sodium or potassium salt of "
        "long-chain fatty acids produced by base hydrolysis of fats or oils.",
        "hint": "What is the chemical identity of soap?",
        "expln_a": "Correct. Soap is precisely defined as the sodium or potassium salt of "
        "long-chain fatty acids. The long hydrocarbon chain is hydrophobic, while the "
        "carboxylate end is hydrophilic — this dual nature is what gives soap its cleaning "
        "power. Base hydrolysis of fats or oils, known as saponification, is the reaction "
        "that produces these salts. The definition is exact and complete.",
        "expln_b": "Incorrect. This is the accurate definition of soap. Soaps are 'sodium "
        "or potassium salts of fatty acids' where fatty acids refer to long-chain "
        "carboxylic acids. The molecule has a hydrophobic hydrocarbon tail and a "
        "hydrophilic carboxylate head. Understanding this dual structure is foundational "
        "to understanding both soap preparation and its cleansing mechanism.",
        "correct_answer": "A",
    },
    {
        "question": "Karthik, a process engineer, explains that saponification is the acid "
        "hydrolysis of triglycerides using hydrochloric acid to produce soap and glycerol.",
        "hint": "Is acid hydrolysis the correct route for soap production?",
        "expln_a": "Incorrect. Saponification is base hydrolysis, not acid hydrolysis. A "
        "fat or oil reacts with a base — typically sodium hydroxide — to produce soap (the "
        "sodium salt of fatty acids) and glycerol as a by-product. Acid hydrolysis of "
        "triglycerides yields free fatty acids and glycerol, not soap salts. Karthik has "
        "confused two different hydrolysis pathways with very different products.",
        "expln_b": "Correct. Karthik is wrong on both counts. Saponification uses a base "
        "such as sodium hydroxide, not an acid. Base hydrolysis is also called "
        "saponification — the reaction produces soap and glycerol. Acid hydrolysis splits "
        "the ester bonds too, but gives free fatty acids rather than the sodium or "
        "potassium salts that constitute soap. The type of reagent determines the product "
        "entirely.",
        "correct_answer": "B",
    },
]

REQUIRED_ITEM_FIELDS = ["question", "hint", "expln_a", "expln_b", "correct_answer"]


def fmt_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def chunk_segments(segments: list[dict], chunk_seconds: int = 150):
    """Group transcript segments into consecutive windows, labeled by each window's end timestamp."""
    if not segments:
        return []
    chunks = []
    window_end = chunk_seconds
    current = []
    for seg in segments:
        current.append(seg)
        if seg["end"] >= window_end:
            chunks.append((seg["end"], current))
            current = []
            window_end = seg["end"] + chunk_seconds
    if current:
        chunks.append((current[-1]["end"], current))
    return [(end, " ".join(s["text"] for s in segs)) for end, segs in chunks]


def _system_prompt(n_questions: int) -> str:
    examples = "\n\n".join(
        f"Example:\nQuestion: {r['question']}\nHint: {r['hint']}\n"
        f"Option A: True | Expln-A: {r['expln_a']}\n"
        f"Option B: False | Expln-B: {r['expln_b']}\n"
        f"Correct Answer: {r['correct_answer']}"
        for r in STYLE_EXAMPLE_ROWS
    )
    return (
        "You write exam-style question-bank items for a lecture transcript segment.\n"
        "Each item is a TRUE/FALSE statement-judgment question: attribute a plausible "
        "statement about the transcript's content to a persona (a student, engineer, "
        "researcher, etc.) — the statement may be entirely correct or contain a subtle "
        "factual error. Option A is always \"True\", Option B is always \"False\". Write a "
        "3-5 sentence explanation for each option that quotes or paraphrases the "
        "transcript, explaining why that option is right or wrong. correct_answer is "
        "exactly one of \"A\" or \"B\".\n\n"
        f"{examples}\n\n"
        f"Generate exactly {n_questions} such questions strictly from the transcript "
        "segment given by the user, covering different points in it. Respond with a JSON "
        "object of exactly this shape and nothing else: "
        '{"questions": [{"question": "...", "hint": "...", "expln_a": "...", '
        '"expln_b": "...", "correct_answer": "A"}, ...]}'
    )


def _validate_questions(items, n_questions):
    valid = [
        item for item in items
        if isinstance(item, dict)
        and all(field in item for field in REQUIRED_ITEM_FIELDS)
        and item["correct_answer"] in ("A", "B")
    ]
    if not valid:
        raise ValueError("no valid question items in model response")
    return valid[:n_questions]


def generate_questions_for_chunk(chunk_text: str, n_questions: int, client, model: str):
    messages = [
        {"role": "system", "content": _system_prompt(n_questions)},
        {"role": "user", "content": f"Transcript segment:\n{chunk_text}"},
    ]
    last_err = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            data = json.loads(resp.choices[0].message.content)
            items = data.get("questions") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError("model response missing a 'questions' array")
            return _validate_questions(items, n_questions)
        except Exception as e:
            last_err = e
            log.warning(f"  chunk generation attempt {attempt + 1} failed: {e}")
    raise RuntimeError(f"question generation failed after retries: {last_err}")


def generate_question_bank(
    segments: list[dict],
    template_columns: list[str],
    client,
    model: str,
    questions_per_segment: int = 5,
    chunk_seconds: int = 150,
    progress_cb=None,
) -> list[dict]:
    chunks = chunk_segments(segments, chunk_seconds)
    rows = []
    sno = 1
    for idx, (end_ts, chunk_text) in enumerate(chunks, start=1):
        questions = generate_questions_for_chunk(
            chunk_text, questions_per_segment, client, model
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
        if progress_cb:
            progress_cb(idx, len(chunks))
    return rows


def rows_to_csv_str(rows: list[dict], template_columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(template_columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in template_columns])
    return buf.getvalue()
