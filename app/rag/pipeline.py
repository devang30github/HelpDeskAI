"""Baseline RAG pipeline: retrieve -> build prompt -> generate -> parse.

Retrieval, prompting and generation are injected, so Version 2 swaps components
(retriever, prompts, LLM) without touching this orchestration or the ticket system.
"""
import re
import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.rag.generation import LLMClient, get_llm
from app.rag.prompts import INSUFFICIENT_MARKER, PROMPT_VERSION, build_messages
from app.rag.retrieval import Retriever, get_retriever
from app.rag.types import RetrievedChunk
from app.rag.vector_store import read_manifest

NO_EVIDENCE_MESSAGE = (
    "I could not find enough information in the knowledge base to answer that."
)
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_INSUFFICIENT_RE = re.compile(r"^\W*INSUFFICIENT[_ ]EVIDENCE\W*", re.IGNORECASE)


@dataclass
class SourceRef:
    index: int  # the [n] number used in the prompt and answer
    chunk_id: str
    doc_id: str
    source_name: str
    title: str
    score: float
    rank: int
    cited: bool  # True only if the answer actually contains [index]


@dataclass
class RAGResult:
    question: str
    answer: str
    answerable: bool
    sources: list[SourceRef]
    retrieved: list[RetrievedChunk] = field(repr=False)
    timings: dict = field(default_factory=dict)  # retrieval_ms, generation_ms, total_ms
    usage: dict = field(default_factory=dict)    # prompt/completion/total tokens
    config: dict = field(default_factory=dict)

    @property
    def contexts(self) -> list[str]:
        return [c.text for c in self.retrieved]


def parse_answer(raw: str, n_sources: int) -> tuple[str, bool, set[int]]:
    """Returns (display_text, answerable, cited_indices)."""
    text = raw.strip()
    answerable = True
    match = _INSUFFICIENT_RE.match(text)
    if match:
        answerable = False
        text = text[match.end():].strip() or NO_EVIDENCE_MESSAGE
    cited: set[int] = set()
    for group in _CITATION_RE.findall(text):
        for part in group.split(","):
            n = int(part.strip())
            if 1 <= n <= n_sources:
                cited.add(n)
    return text, answerable, cited


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int,
        extra_config: dict | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self._extra_config = extra_config or {}

    def config(self) -> dict:
        llm_config = self.llm.config() if hasattr(self.llm, "config") else {"model": self.llm.model}
        return {
            "pipeline": "baseline_rag_v1",
            "prompt_version": PROMPT_VERSION,
            "retriever": getattr(self.retriever, "name", type(self.retriever).__name__),
            "top_k": self.top_k,
            "llm": llm_config,
            **self._extra_config,
        }

    def answer(self, question: str, history: list[dict] | None = None) -> RAGResult:
        total_start = time.perf_counter()

        t = time.perf_counter()
        chunks = self.retriever.retrieve(question, self.top_k)
        retrieval_ms = (time.perf_counter() - t) * 1000

        if not chunks:  # empty index: nothing to ground an answer on, skip the LLM call
            return RAGResult(
                question=question,
                answer=NO_EVIDENCE_MESSAGE,
                answerable=False,
                sources=[],
                retrieved=[],
                timings={"retrieval_ms": retrieval_ms, "generation_ms": 0.0,
                         "total_ms": (time.perf_counter() - total_start) * 1000},
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                config=self.config(),
            )

        messages = build_messages(question, chunks, history)
        llm_result = self.llm.complete(messages)
        text, answerable, cited = parse_answer(llm_result.text, len(chunks))

        sources = [
            SourceRef(
                index=i,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_name=c.source_name,
                title=c.metadata.get("title") or c.doc_id,
                score=c.score,
                rank=c.rank,
                cited=i in cited,
            )
            for i, c in enumerate(chunks, start=1)
        ]
        return RAGResult(
            question=question,
            answer=text,
            answerable=answerable,
            sources=sources,
            retrieved=chunks,
            timings={
                "retrieval_ms": retrieval_ms,
                "generation_ms": llm_result.latency_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000,
            },
            usage={
                "prompt_tokens": llm_result.prompt_tokens,
                "completion_tokens": llm_result.completion_tokens,
                "total_tokens": llm_result.total_tokens,
            },
            config=self.config(),
        )


def get_pipeline() -> RAGPipeline:
    """Builds a pipeline from settings. Retriever and LLM client are cached,
    so this is cheap and always sees a freshly rebuilt index."""
    settings = get_settings()
    retriever = get_retriever()
    manifest = read_manifest(settings.index_dir)
    extra = {
        "embedding_model": manifest.get("embedding_model"),
        "chunk_size": manifest.get("chunk_size"),
        "chunk_overlap": manifest.get("chunk_overlap"),
        "index_built_at": manifest.get("built_at"),
        "kb_files": manifest.get("files"),  # file -> sha256, pins the exact KB version
    }
    return RAGPipeline(retriever, get_llm(), settings.retrieval_top_k, extra)