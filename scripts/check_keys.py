"""Check every configured Groq API key:  python -m scripts.check_keys
Makes one tiny call per key and prints Groq's rate-limit counters (keys are never printed).
If the 'remaining' counters keep dropping from one key to the next instead of each starting
fresh, the keys share one rate-limit pool and rotating them gains nothing."""
import sys

from openai import OpenAI

from app.core.config import get_settings

SHOW = ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens")


def main() -> int:
    s = get_settings()
    keys = s.groq_key_list()
    if not keys:
        print("No keys configured. Set GROQ_API_KEYS=key1,key2,... in .env")
        return 1
    print(f"{len(keys)} key(s) configured | model {s.groq_model}\n")
    failures = 0
    for number, key in enumerate(keys, start=1):
        client = OpenAI(api_key=key, base_url=s.groq_base_url, max_retries=0, timeout=30)
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=s.groq_model, messages=[{"role": "user", "content": "Reply with the word ok."}],
                max_completion_tokens=64)
        except Exception as e:  # print the type and status only, never the message (it can echo key fragments)
            failures += 1
            print(f"key-{number}: FAILED  {type(e).__name__} (HTTP {getattr(e, 'status_code', 'n/a')})")
            continue
        info = {k.replace("x-ratelimit-", ""): raw.headers.get(k) for k in SHOW if raw.headers.get(k) is not None}
        print(f"key-{number}: ok  {info}")
    print(f"\n{len(keys) - failures}/{len(keys)} keys working.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())