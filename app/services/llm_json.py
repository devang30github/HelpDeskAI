import json


def extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates code fences and prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None