"""Claude-powered helpers for the Ask XMarket page.

Claude never invents prices: it only turns a plain-English request into search filters, and turns
our model's numbers for one home into a short plain-English explanation.
"""
import hashlib
import json
import os

import anthropic

from common import ROOT

MODEL = "claude-opus-5-5"
CACHE_PATH = ROOT / "reports" / "ai_cache.json"  # answers survive restarts, so a demo works on bad Wi-Fi
HOME_TYPES = ["SINGLE_FAMILY", "TOWNHOUSE", "CONDO", "MULTI_FAMILY", "MANUFACTURED"]
HUBS = ["downtown", "tmc", "uh", "galleria", "energy_corridor", "none"]


def _client():
    """Use ANTHROPIC_API_KEY from the environment, else from the project's .env file (never printed)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = ROOT / ".env"
        for line in env.read_text().splitlines() if env.exists() else []:
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                return anthropic.Anthropic(api_key=value.strip().strip("\"'"))
    return anthropic.Anthropic()


class AIUnavailable(Exception):
    """Claude couldn't answer (offline, rate-limited, or declined)."""


def _nullable(t):
    return {"type": [t, "null"]}


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string", "description": "One plain-English sentence restating the search"},
        "annual_income": _nullable("number"),
        "max_monthly_cost": _nullable("number"),
        "max_price": _nullable("number"),
        "min_bedrooms": _nullable("integer"),
        "min_bathrooms": _nullable("number"),
        "min_sqft": _nullable("number"),
        "home_types": {"type": "array", "items": {"type": "string", "enum": HOME_TYPES}},
        "near": {"type": "string", "enum": HUBS},
        "max_miles": _nullable("number"),
        "zip_codes": {"type": "array", "items": {"type": "string"}},
        "cities": {"type": "array", "items": {"type": "string"}},
        "new_construction": {"type": "string", "enum": ["only", "exclude", "any"]},
        "built_after": _nullable("integer"),
        "deals_only": {"type": "boolean"},
        "sort_by": {"type": "string", "enum": ["best_deal", "lowest_price", "lowest_monthly_cost", "largest", "newest"]},
    },
    "required": ["interpretation", "annual_income", "max_monthly_cost", "max_price", "min_bedrooms", "min_bathrooms",
                 "min_sqft", "home_types", "near", "max_miles", "zip_codes", "cities", "new_construction",
                 "built_after", "deals_only", "sort_by"],
    "additionalProperties": False,
}

SEARCH_SYSTEM = """You turn a Houston home buyer's request into search filters for a database of 19,430 homes listed for sale in Greater Houston (June 2024).

Rules:
- Use null / empty list / "any" / "none" for anything the buyer did not ask for. Never invent constraints.
- "near" is one of: downtown, tmc (Texas Medical Center), uh (University of Houston), galleria (Galleria/Uptown), energy_corridor. If they name a place near a hub without a distance, use max_miles 10.
- Map Houston neighborhoods to ZIP codes or cities from the lists below (e.g. Heights -> 77008, Montrose -> 77006/77019, Midtown -> 77004/77002, Memorial -> 77024, Clear Lake -> 77058/77062, The Woodlands -> cities ["The Woodlands", "Spring"]).
- "annual_income": their household income if they state one. If they give a monthly budget, use max_monthly_cost (all-in: mortgage + tax + insurance).
- "deals_only": true only if they ask for bargains, deals, or underpriced homes. Default sort is best_deal.
- "house" means SINGLE_FAMILY; "condo", "townhome" map to their types. Family of N usually implies at least N-1 bedrooms only if they say so; otherwise leave null.
- "interpretation": restate the search in one short, friendly sentence.

Cities in the data: {cities}
ZIP codes in the data: {zips}"""

EXPLAIN_SYSTEM = """You are XMarket, a friendly and careful Houston housing guide for first-time buyers.
You receive one home's facts and our machine-learning model's numbers. Write 4 short paragraphs in plain English (under 170 words total):
1. Verdict: is the asking price below, within, or above our fair-value range, by how much?
2. Why: the 2-3 biggest price drivers, in everyday words.
3. Real cost: the all-in monthly payment and the income it takes.
4. One concrete next step and one caution (e.g. inspection, flood history, HOA).
Use only the numbers provided. Do not give personalized financial or investment advice. No headings, no bullet lists."""


def _load_cache():
    try:
        return json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _call(kind, system, user, schema=None, max_tokens=4000):
    key = hashlib.sha256(json.dumps([kind, MODEL, system, user]).encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]
    output_config = {"effort": "low"}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        resp = _client().beta.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], output_config=output_config,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.CredentialsError as e:
        raise AIUnavailable("No Anthropic API key found (set ANTHROPIC_API_KEY or add it to .env).") from e
    except anthropic.APIConnectionError as e:
        raise AIUnavailable("No internet connection to Claude.") from e
    except anthropic.RateLimitError as e:
        raise AIUnavailable("Claude is rate-limited right now. Try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise AIUnavailable(f"Claude API error ({e.status_code}).") from e
    if resp.stop_reason == "refusal":
        raise AIUnavailable("Claude declined this request.")
    if resp.stop_reason == "max_tokens":
        raise AIUnavailable("Claude's answer was cut off. Try a shorter request.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    result = json.loads(text) if schema else text.strip()
    cache[key] = result
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return result


def parse_search(query, cities, zips):
    system = SEARCH_SYSTEM.format(cities=", ".join(cities), zips=", ".join(zips))
    return _call("search", system, query, schema=SEARCH_SCHEMA)


def explain_home(facts: dict):
    return _call("explain", EXPLAIN_SYSTEM, json.dumps(facts, indent=1), max_tokens=6000)
