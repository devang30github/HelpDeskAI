from app.rag.generation import LLMResult
from app.rag.pipeline import NO_EVIDENCE_MESSAGE, RAGPipeline, parse_answer
from app.rag.types import RetrievedChunk


def _chunk(i, doc="vpn-guide"):
    return RetrievedChunk(f"{doc}::chunk-00{i}", doc, f"{doc}.pdf", f"text {i}",
                          0.9 - i * 0.1, i + 1, {"title": "VPN Guide"})


class FakeRetriever:
    name = "fake"

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, k):
        self.calls.append((query, k))
        return self.chunks[:k]


class FakeLLM:
    model = "fake-model"

    def __init__(self, text):
        self.text = text
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return LLMResult(self.text, self.model, 100, 20, 120, 5.0, "stop")

    def config(self):
        return {"model": self.model}


def test_answer_with_citations():
    retriever, llm = FakeRetriever([_chunk(0), _chunk(1), _chunk(2)]), FakeLLM("Restart the client [1][2].")
    result = RAGPipeline(retriever, llm, top_k=3).answer("VPN fails")
    assert result.answerable is True
    assert [s.cited for s in result.sources] == [True, True, False]
    assert retriever.calls == [("VPN fails", 3)]
    assert result.usage["total_tokens"] == 120
    assert set(result.timings) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert result.config["retriever"] == "fake" and result.config["top_k"] == 3
    assert result.contexts == ["text 0", "text 1", "text 2"]


def test_insufficient_marker_sets_answerable_false():
    llm = FakeLLM("INSUFFICIENT_EVIDENCE: The documents do not cover payroll.")
    result = RAGPipeline(FakeRetriever([_chunk(0)]), llm, top_k=2).answer("payroll?")
    assert result.answerable is False
    assert result.answer == "The documents do not cover payroll."
    assert not any(s.cited for s in result.sources)


def test_marker_only_falls_back_to_default_message():
    result = RAGPipeline(FakeRetriever([_chunk(0)]), FakeLLM("INSUFFICIENT_EVIDENCE"), top_k=1).answer("q")
    assert result.answerable is False and result.answer == NO_EVIDENCE_MESSAGE


def test_empty_retrieval_skips_the_llm():
    llm = FakeLLM("should not be used")
    result = RAGPipeline(FakeRetriever([]), llm, top_k=3).answer("q")
    assert result.answerable is False and result.sources == []
    assert llm.messages is None
    assert result.timings["generation_ms"] == 0.0


def test_out_of_range_citation_is_ignored():
    result = RAGPipeline(FakeRetriever([_chunk(0), _chunk(1)]), FakeLLM("Do this [7]."), top_k=2).answer("q")
    assert not any(s.cited for s in result.sources)


def test_history_is_passed_to_the_llm():
    llm = FakeLLM("ok [1]")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    RAGPipeline(FakeRetriever([_chunk(0)]), llm, top_k=1).answer("q", history=history)
    assert [m["role"] for m in llm.messages] == ["system", "user", "assistant", "user"]


def test_parse_answer_variants():
    assert parse_answer("insufficient evidence - nothing found", 2) == ("nothing found", False, set())
    text, answerable, cited = parse_answer("Do A [1, 2] then B [3].", 3)
    assert answerable and cited == {1, 2, 3}