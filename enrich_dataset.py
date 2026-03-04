import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq

BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "scrapper" / "output" / "shl_individual_tests.json"
OUTPUT_PATH = BASE_DIR / "scrapper" / "output" / "shl_individual_tests_enrich.json"

MODEL        = "llama-3.1-8b-instant"
RPM          = 28
INTERVAL     = 60.0 / RPM
RETRY_LIMIT  = 5
RETRY_DELAY  = 15


def init_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def _is_trivial_description(name: str, description: str) -> bool:
    if not description:
        return True
    cleaned = description.strip().lower()
    return cleaned == name.strip().lower() or len(cleaned) < 40


def build_prompt(assessment: dict) -> str:
    name           = assessment.get("name", "")
    description    = assessment.get("description", "").strip()
    test_types     = ", ".join(assessment.get("test_type", []))
    duration       = assessment.get("duration") or 0
    remote_testing = assessment.get("remote_testing", "")
    adaptive       = assessment.get("adaptive_support", "")
    duration_str   = f"{duration} minutes" if duration > 0 else "Not specified"

    desc_block = (
        "(No meaningful description — infer from assessment name and test type)"
        if _is_trivial_description(name, description)
        else description[:800]
    )

    return f"""You are an expert HR consultant familiar with SHL psychometric assessments.

Given the following SHL assessment details, write 2-3 sentences describing:
1. What job roles and seniority levels this assessment is best suited for
2. What specific skills, competencies, or industries it maps to
3. When an employer would typically use this assessment in hiring

Rules:
- Mention concrete job titles specific to THIS assessment (e.g. ".NET developer" not just "software engineer")
- Name industries that specifically USE this skill — not generic lists like "finance, healthcare, technology"
- Focus on what makes THIS assessment unique vs others of the same type
- Do NOT repeat the assessment name
- Do NOT use filler phrases like "This assessment is best suited for"

Assessment Name:  {name}
Test Type:        {test_types}
Duration:         {duration_str}
Remote Testing:   {remote_testing}
Adaptive Support: {adaptive}
Description:      {desc_block}

Return only the 2-3 sentences. No preamble, no bullet points."""


def generate(client: Groq, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def generate_role_summary(client: Groq, assessment: dict) -> str:
    prompt = build_prompt(assessment)
    delay  = RETRY_DELAY

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return generate(client, prompt)
        except Exception as e:
            err      = str(e).lower()
            is_quota = any(w in err for w in ("rate", "quota", "429", "limit", "exhausted"))
            wait     = delay * (2 ** (attempt - 1)) if is_quota else delay
            print(f"\n  [{'Quota' if is_quota else 'Error'}] Attempt {attempt}/{RETRY_LIMIT} — waiting {wait}s... ({e})")
            time.sleep(wait)

    print(f"\n  Skipping '{assessment.get('name')}' after {RETRY_LIMIT} attempts.")
    return ""


def enrich(input_path: Path, output_path: Path):
    print(f"Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    already_done = sum(1 for a in assessments if a.get("role_summary"))
    remaining    = len(assessments) - already_done
    eta          = remaining * INTERVAL / 60

    print(f"\nModel     : {MODEL}")
    print(f"Total     : {len(assessments)}  |  Done: {already_done}  |  Remaining: {remaining}")
    print(f"Rate      : {RPM} RPM ({INTERVAL:.1f}s between calls)")
    print(f"ETA       : ~{eta:.1f} minutes\n")

    client = init_client()

    for assessment in tqdm(assessments, desc="Enriching"):
        if assessment.get("role_summary"):
            continue

        assessment["role_summary"] = generate_role_summary(client, assessment)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(assessments, f, ensure_ascii=False, indent=2)

        time.sleep(INTERVAL)

    print(f"\nDone. Saved to: {output_path}")


if __name__ == "__main__":
    enrich(INPUT_PATH, OUTPUT_PATH)