"""RAGAS wiring (ragas 0.4 collections API): an explicit judge LLM and a separately pinned
embedding model. One judge client per API key, so the evaluation load is spread across keys.
Isolated here so a RAGAS version change only touches this file."""
import asyncio
import importlib.metadata as md

from app.core.config import Settings, get_settings

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _local_embeddings(embedder):
    """Adapter: lets RAGAS use a local sentence-transformers model (Groq has no embeddings endpoint)."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalEmbeddings(BaseRagasEmbedding):
        def __init__(self):
            super().__init__()

        def embed_text(self, text: str, **kwargs):
            return embedder.encode([text])[0].tolist()

        async def aembed_text(self, text: str, **kwargs):
            return await asyncio.to_thread(self.embed_text, text)

    return LocalEmbeddings()


def _metrics_for_key(s: Settings, api_key: str, embeddings) -> dict:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    client = AsyncOpenAI(api_key=api_key, base_url=s.groq_base_url, timeout=s.ragas_timeout_seconds,
                         max_retries=s.groq_max_retries)
    judge = llm_factory(s.ragas_judge_model, client=client, temperature=0, max_tokens=s.ragas_max_tokens)
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def build_metrics_pool(settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    keys = s.groq_key_list()
    if not keys:
        raise RuntimeError("no Groq API key configured (GROQ_API_KEY or GROQ_API_KEYS); the RAGAS judge needs one")
    from app.rag.embeddings import get_embedder

    embeddings = _local_embeddings(get_embedder(s.ragas_embedding_model))
    return [_metrics_for_key(s, key, embeddings) for key in keys]


def build_metrics(settings: Settings | None = None) -> dict:
    """A single set of metrics (the first key); used by scripts.check_ragas."""
    return build_metrics_pool(settings)[0]


def describe_judge(settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    try:
        version = md.version("ragas")
    except md.PackageNotFoundError:
        version = None
    return {
        "ragas_version": version,
        "judge_model": s.ragas_judge_model,
        "judge_base_url": s.groq_base_url,
        "judge_embedding_model": s.ragas_embedding_model,
        "judge_max_tokens": s.ragas_max_tokens,
        "judge_api_keys": len(s.groq_key_list()),
        "same_family_as_generator": s.ragas_judge_model.split("/")[0] == s.groq_model.split("/")[0],
    }