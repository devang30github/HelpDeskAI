"""Verify the RAGAS wiring before any evaluation:  python -m scripts.check_ragas
Makes a handful of small judge calls and checks that obviously good inputs score
higher than obviously bad ones."""
import asyncio
import importlib.metadata as md
import platform
import sys
import time

from app.core.config import get_settings
from app.evaluation.ragas_setup import build_metrics, describe_judge

Q = "What is the capital of France?"
GOOD_CTX = ["Paris is the capital and largest city of France."]
BAD_CTX = ["Bananas are rich in potassium and are usually yellow."]
REF = "Paris is the capital of France."

HINTS = """
Troubleshooting (reported by third parties for 2026, not verified by me):
  - run `uv pip check` to see conflicting packages
  - ragas 0.4.3 reportedly fails to import with langchain-community >= 0.4.2: pin langchain-community below it
  - instructor 1.16 reportedly requires openai < 3: try `uv pip install "openai<3"`
  - if the error is about structured output or response format, paste the full traceback so I can adapt the wiring
"""


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


async def _score(metric, **kwargs) -> float:
    result = await metric.ascore(**kwargs)
    return float(getattr(result, "value", result))


async def main_async() -> int:
    try:
        metrics = build_metrics()
    except Exception as e:
        print(f"Could not build the RAGAS metrics: {type(e).__name__}: {e}")
        print(HINTS)
        return 1

    checks = [
        ("faithfulness", "faithful vs unfaithful",
         dict(user_input=Q, retrieved_contexts=GOOD_CTX, response="The capital of France is Paris."),
         dict(user_input=Q, retrieved_contexts=GOOD_CTX, response="The capital of France is Lyon, home to two million people.")),
        ("answer_relevancy", "on-topic vs off-topic",
         dict(user_input=Q, response="The capital of France is Paris."),
         dict(user_input=Q, response="Bananas are rich in potassium.")),
        ("context_precision", "relevant vs irrelevant context",
         dict(user_input=Q, retrieved_contexts=GOOD_CTX, reference=REF),
         dict(user_input=Q, retrieved_contexts=BAD_CTX, reference=REF)),
        ("context_recall", "context has the answer vs not",
         dict(user_input=Q, retrieved_contexts=GOOD_CTX, reference=REF),
         dict(user_input=Q, retrieved_contexts=BAD_CTX, reference=REF)),
    ]
    failures, calls, started = 0, 0, time.perf_counter()
    for name, label, good_kwargs, bad_kwargs in checks:
        try:
            good = await _score(metrics[name], **good_kwargs)
            bad = await _score(metrics[name], **bad_kwargs)
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            continue
        calls += 2
        verdict = "ok" if good > bad else "UNEXPECTED (good <= bad)"
        failures += good <= bad
        print(f"{'PASS' if good > bad else 'FAIL'} {name:<18} {label}: {good:.2f} vs {bad:.2f}  [{verdict}]")

    elapsed = time.perf_counter() - started
    if calls:
        print(f"\n{calls} metric calls in {elapsed:.1f}s ({elapsed / calls:.1f}s per call, includes several judge requests each)")
    if failures:
        print(HINTS)
    return 1 if failures else 0


def main() -> int:
    s = get_settings()
    print(f"python {platform.python_version()} | ragas {_version('ragas')} | openai {_version('openai')} "
          f"| instructor {_version('instructor')}")
    print(f"judge model: {s.ragas_judge_model} | generator: {s.groq_model} | embeddings: {s.ragas_embedding_model}")
    print(f"judge details: {describe_judge(s)}\n")
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())