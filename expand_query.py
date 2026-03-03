import re
from typing import List
import google.generativeai as genai


# Only expand queries longer than this — short queries don't need it
EXPANSION_WORD_THRESHOLD = 30


def _word_count(text: str) -> int:
    return len(text.split())


def expand_query(query: str, gemini_model) -> List[str]:
    
    if _word_count(query) < EXPANSION_WORD_THRESHOLD:
        return [query]

    prompt = f"""You are helping build a search system for HR assessments.

Given this job description, extract 3-5 SHORT search queries (5-10 words each).
Each query should focus on a DIFFERENT skill or competency area mentioned in the JD.
Queries should use keywords that would appear in assessment names or descriptions.

Rules:
- Cover technical skills separately from soft skills
- Cover cognitive/reasoning separately from personality/behaviour  
- Use concrete skill names (e.g. "Java programming", "SQL database", "verbal reasoning")
- Do NOT repeat the same skill across queries
- Keep each query under 10 words

JOB DESCRIPTION:
{query[:2000]}

Respond with ONLY a JSON array of query strings.
Example: ["Java programming OOP", "verbal reasoning communication", "teamwork collaboration personality"]

Return only the JSON array, nothing else."""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        sub_queries = __import__("json").loads(raw)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            # Always include a compressed version of the original as a safety net
            sub_queries.append(_compress_query(query))
            return sub_queries[:6]  # cap at 6 sub-queries
    except Exception:
        pass

    return _rule_based_expansion(query)


def _compress_query(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
    first = lines[0] if lines else text[:100]
    words = first.split()
    return " ".join(words[:15])


def _rule_based_expansion(query: str) -> List[str]:
    sub_queries = []

    # Common tech skills
    tech_skills = re.findall(
        r'\b(java|python|sql|javascript|html|css|selenium|excel|tableau|'
        r'react|angular|node|aws|azure|docker|kubernetes|c\+\+|\.net|'
        r'machine learning|data analysis|power bi|seo|drupal)\b',
        query.lower()
    )
    if tech_skills:
        sub_queries.append(" ".join(dict.fromkeys(tech_skills[:5])))  # deduplicated

    soft_signals = re.findall(
        r'\b(communication|leadership|teamwork|collaboration|interpersonal|'
        r'management|personality|culture|analytical|reasoning|verbal|numerical)\b',
        query.lower()
    )
    if soft_signals:
        sub_queries.append(" ".join(dict.fromkeys(soft_signals[:5])))

    sub_queries.append(_compress_query(query))
    return sub_queries if sub_queries else [query]