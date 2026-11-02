from app.rag.bm25 import BM25Index, tokenize
from app.rag.types import Chunk


def _chunk(i, text, title=""):
    return Chunk(chunk_id=f"d{i}::chunk-000", doc_id=f"d{i}", source_name=f"d{i}.txt", chunk_index=0,
                 text=text, metadata={"title": title})


def test_tokenizer_keeps_codes_whole_and_adds_parts():
    tokens = tokenize("Error VPN-612 on Wi-Fi")
    assert {"vpn-612", "vpn", "612", "wi-fi", "wifi"} <= set(tokens)


def test_stopwords_dropped_and_light_stemming_applied():
    assert tokenize("the printers are printing") == ["printer", "print"]
    assert tokenize("batteries") == ["battery"]
    assert tokenize("access status") == ["access", "status"]  # -ss and -us endings are kept


def test_exact_code_beats_a_near_miss():
    index = BM25Index([_chunk(0, "APP-381 means the application exits during launch."),
                       _chunk(1, "APP-318 means the catalogue reports an installation rollback.")])
    hits = index.search("what is APP-318", 2)
    assert hits[0][0].chunk_id == "d1::chunk-000" and hits[0][1] > 0


def test_rare_terms_outweigh_common_ones():
    index = BM25Index([_chunk(0, "restart the device and restart the client"),
                       _chunk(1, "restart the device then check the quarantine portal"),
                       _chunk(2, "restart the device and wait")])
    assert index.search("restart quarantine", 3)[0][0].chunk_id == "d1::chunk-000"


def test_no_overlap_empty_query_and_empty_index():
    index = BM25Index([_chunk(0, "restart the client")])
    assert index.search("zzzz qqqq", 3) == [] and index.search("", 3) == [] and index.search("restart", 0) == []
    assert BM25Index([]).search("restart", 3) == []


def test_ties_are_broken_by_chunk_order():
    index = BM25Index([_chunk(0, "reset the token"), _chunk(1, "reset the token")])
    assert [c.chunk_id for c, _ in index.search("reset token", 2)] == ["d0::chunk-000", "d1::chunk-000"]


def test_the_title_is_searchable():
    index = BM25Index([_chunk(0, "Check the cable ends.", title="Laptop Docking Guide"),
                       _chunk(1, "Check the cable ends.", title="Printer Guide")])
    hits = index.search("docking", 2)
    assert [c.chunk_id for c, _ in hits] == ["d0::chunk-000"]