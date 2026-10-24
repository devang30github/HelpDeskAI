"""RAGAS wiring (ragas 0.4 collections API): an explicit judge LLM and a separately pinned
embedding model. Isolated here so a RAGAS version change only touches this file."""
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


def build_metrics(settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (the RAGAS judge uses it)")

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from app.rag.embeddings import get_embedder

    client = AsyncOpenAI(
        api_key=s.groq_api_key,
        base_url=s.groq_base_url,
        timeout=s.ragas_timeout_seconds,
        max_retries=s.groq_max_retries,
    )
    judge = llm_factory(s.ragas_judge_model, client=client, temperature=0, max_tokens=s.ragas_max_tokens)
    embeddings = _local_embeddings(get_embedder(s.ragas_embedding_model))
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


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
        "same_family_as_generator": s.ragas_judge_model.split("/")[0] == s.groq_model.split("/")[0],
    }