"""Ask the RAG pipeline a question.   python -m scripts.ask "My VPN times out, what now?" """
import argparse
import sys

from app.rag.generation import LLMConfigError, LLMError
from app.rag.pipeline import get_pipeline
from app.rag.vector_store import IndexNotFound


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    try:
        result = get_pipeline().answer(args.question)
    except IndexNotFound:
        print("No index found. Run: python -m scripts.ingest")
        return 1
    except LLMConfigError as e:
        print(f"Configuration problem: {e}")
        return 1
    except LLMError as e:
        print(f"LLM error: {e}")
        return 1

    print(f"\nanswerable: {result.answerable}\n")
    print(result.answer)
    print("\nSources (* = cited in the answer):")
    for s in result.sources:
        print(f"  [{s.index}]{'*' if s.cited else ' '} {s.source_name}  {s.chunk_id}  score={s.score:.3f}")
    t, u = result.timings, result.usage
    print(f"\nretrieval {t['retrieval_ms']:.0f} ms | generation {t['generation_ms']:.0f} ms | "
          f"tokens {u['total_tokens']} (prompt {u['prompt_tokens']}, completion {u['completion_tokens']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())