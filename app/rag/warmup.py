"""Load the retriever (FAISS index + embedding model) at server start, so the first
user query does not pay the cold-start cost."""
import logging
import time

from app.rag.generation import LLMError, get_llm
from app.rag.retrieval import get_retriever
from app.rag.vector_store import IndexCorrupt, IndexMismatch, IndexNotFound

logger = logging.getLogger(__name__)


def warm_up() -> dict:
    start = time.perf_counter()
    try:
        retriever = get_retriever()
        t = time.perf_counter()
        retriever.retrieve("warm-up query", 1)  # loads the model and runs one real search
        first_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        retriever.retrieve("second warm-up query", 1)
        steady_ms = (time.perf_counter() - t) * 1000
    except (IndexNotFound, IndexMismatch, IndexCorrupt) as e:
        logger.warning("Warm-up skipped: %s", e)
        return {"warmed": False, "reason": type(e).__name__}
    except Exception:  # never block startup (for example, no network on the first model download)
        logger.exception("Warm-up failed (non-fatal)")
        return {"warmed": False, "reason": "error"}

    try:
        get_llm()  # builds the client object only; no network call
    except LLMError as e:
        logger.info("LLM client not configured: %s", e)

    total_ms = (time.perf_counter() - start) * 1000
    logger.info("Warm-up finished in %.0f ms (first retrieval %.0f ms, steady-state %.0f ms)",
                total_ms, first_ms, steady_ms)
    return {"warmed": True, "ms": total_ms}