from __future__ import annotations
import json
import logging
import os
from openai import OpenAI
from schemas import FilterSchema
from config import INTENT_MODEL

_client: OpenAI | None = None
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a travel intent extractor for a hotel/property search engine.

Parse the user's natural language travel query into a structured set of search filters.

Rules:
- destination: extract city/region if mentioned (e.g. "Paris", "Bali coast", "near Amalfi")
- vibe: pick from [romantic, adventure, family, wellness, party, luxury, budget, cultural, quiet, scenic] — can be multiple
- amenities: extract specific amenities, including IMPLICIT ones:
    "sunset views" → ["scenic views", "balcony or terrace"]
    "good gym" → ["gym", "fitness center"]
    "infinity pool" → ["swimming pool", "infinity pool"]
    "pet friendly" → ["pet friendly"]
- property_type: pick from [hotel, boutique, villa, resort] — can be multiple
- budget_range: (min, max) per night in USD if budget is mentioned (e.g. "under $200" → (0, 200), "around $300" → (250, 350))
- special_needs: any accessibility, crib, parking, pet, or specific requirements
- location_vibe: one of [near beach, city center, countryside, waterfront, riverside, hilltop, old town, beachfront] if mentioned

Be generous with vibe and amenity inference — users rarely say exactly what they want."""

_JSON_INSTRUCTIONS = """
Return a single JSON object with exactly these keys:
- destination
- vibe
- amenities
- property_type
- budget_range
- special_needs
- location_vibe

Use null for missing single values, [] for missing lists, and keep budget_range as [min, max] when present.
Do not wrap the JSON in markdown fences.
""".strip()


def _normalize_filter_schema_payload(data: dict) -> dict:
    normalized = dict(data or {})

    for field in ("vibe", "amenities", "property_type", "special_needs"):
        value = normalized.get(field)
        if value is None:
            normalized[field] = []
        elif isinstance(value, str):
            normalized[field] = [value]
        elif not isinstance(value, list):
            normalized[field] = list(value) if isinstance(value, tuple) else []

    budget = normalized.get("budget_range")
    if budget in (None, "", []):
        normalized["budget_range"] = None
    elif isinstance(budget, tuple):
        normalized["budget_range"] = list(budget)

    for field in ("destination", "location_vibe"):
        value = normalized.get(field)
        if value in ("", [], {}):
            normalized[field] = None
        elif isinstance(value, list):
            normalized[field] = value[0] if value else None

    return normalized


def _parse_filter_schema_from_content(content: str) -> FilterSchema:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = _normalize_filter_schema_payload(json.loads(text or "{}"))
    return FilterSchema.model_validate(payload)


def extract_intent(user_query: str) -> FilterSchema:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = _client.chat.completions.create(
        model=INTENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "system", "content": _JSON_INSTRUCTIONS},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return _parse_filter_schema_from_content(content)
    except Exception as exc:
        logger.error("Intent parser returned invalid JSON: %s", content)
        raise ValueError(f"Invalid intent JSON: {exc}") from exc
