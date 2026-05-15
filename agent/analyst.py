# agent/analyst.py

"""
Universal LLM analyst — works with ANY OpenAI-compatible API:
  - Groq (free)        → GROQ_API_KEY + https://api.groq.com/openai/v1
  - OpenRouter (free)   → OPENROUTER_API_KEY + https://openrouter.ai/api/v1
  - Together (free)     → TOGETHER_API_KEY + https://api.together.xyz/v1
  - Ollama (local)      → no key + http://localhost:11434/v1
  - LM Studio (local)   → no key + http://localhost:1234/v1
  - OpenAI              → OPENAI_API_KEY + https://api.openai.com/v1
  - Google Gemini       → GEMINI_API_KEY + https://generativelanguage.googleapis.com/v1beta/openai
  - Mistral             → MISTRAL_API_KEY + https://api.mistral.ai/v1

Just set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL in your .env file.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Universal config ─────────────────────────────────────────────────────────
# ── Universal config ─────────────────────────────────────────────────────────
# Secrets are loaded dynamically inside _call_llm to ensure Streamlit Secrets
# have been injected into os.environ before reading.


def _call_llm(prompt: str) -> str:
    """
    Call any OpenAI-compatible chat completions API.
    This single function works with every provider listed above.
    """
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip()
    api_key = os.getenv("LLM_API_KEY", "ollama").strip()
    model = os.getenv("LLM_MODEL", "qwen2.5:7b").strip()

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI analyst unavailable: {e}]"


def build_prompt(
    similarity_results: list,
    dimension_scores: dict,
    available_indicators: list,
    live_vector,
    query_date: str = "today"
) -> str:
    """Builds the structured prompt for the analyst agent."""
    top_3 = similarity_results[:3]

    indicator_stress = [(available_indicators[i], abs(live_vector[i]))
                        for i in range(len(available_indicators))]
    indicator_stress.sort(key=lambda x: x[1], reverse=True)
    top_indicators = indicator_stress[:5]

    top_analogue_text = "\n".join([
        f"  {i+1}. {r['name']} ({r['short']}): {r['similarity']:.1f}% similarity\n"
        f"     Key signature: {r['key_signature']}\n"
        f"     Peak date: {r['peak_date']}"
        for i, r in enumerate(top_3)
    ])

    dimension_text = "\n".join([
        f"  {dim}: {score:.1f}/100 stress"
        for dim, score in sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    ])

    indicator_text = "\n".join([
        f"  {ind.replace('_', ' ')}: {val:.2f}\u03c3 deviation"
        for ind, val in top_indicators
    ])

    prompt = f"""You are AUTOPSY, a quantitative market risk analyst system. 
Your job is to analyze current market structure and produce a concise, precise risk narrative.

## Current Market Snapshot (as of {query_date})

### Top Crisis Structural Analogues:
{top_analogue_text}

### Stress by Dimension (0-100 scale):
{dimension_text}

### Most Stressed Indicators:
{indicator_text}

## Your Task

Write a structured risk narrative with EXACTLY these four sections:

**STRUCTURAL ASSESSMENT** (2-3 sentences)
Describe what the current market structure fingerprint reveals.

**HISTORICAL ANALOGUES** (3-4 sentences)
Explain what the top 1-2 analogues share with the current fingerprint.

**KEY DIVERGENCES** (2-3 sentences)
What aspects of the current fingerprint explicitly differ from the top analogue?

**RISK POSTURE** (2-3 sentences)
What should a risk-aware institutional investor monitor closely?

Keep the total response under 350 words. Be precise. Write as a senior quant risk officer would brief a CIO."""

    return prompt


def run_analyst(
    similarity_results: list,
    dimension_scores: dict,
    available_indicators: list,
    live_vector,
    query_date: str = "today"
) -> str:
    """Calls the LLM and returns the structured narrative string."""
    prompt = build_prompt(
        similarity_results, dimension_scores, available_indicators, live_vector, query_date
    )

    result = _call_llm(prompt)

    if result.startswith("[AI analyst unavailable"):
        if similarity_results:
            result += f"\n\nTop analogue: {similarity_results[0]['name']} ({similarity_results[0]['similarity']:.1f}% similarity)"
    return result
