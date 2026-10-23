import pytest

from app.core.config import BASE_DIR
from app.rag.embeddings import get_embedder
from app.rag.retrieval import DenseRetriever
from app.rag.vector_store import MANIFEST_FILE, VectorStore

REAL_INDEX = BASE_DIR / "data" / "index"

pytestmark = pytest.mark.skipif(
    not (REAL_INDEX / MANIFEST_FILE).exists(),
    reason="build the index first: python -m scripts.ingest",
)


@pytest.fixture(scope="module")
def retriever():
    store = VectorStore.load(REAL_INDEX)
    return DenseRetriever(store, get_embedder(store.manifest["embedding_model"]))


@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("How do I reset my password?", "account-lockout-and-signin-guide"),
        ("My laptop battery looks swollen", "swollen-battery-device-safety"),
        ("The printer shows offline", "printer-queue-and-paper-jam-guide"),
    ],
)

def test_expected_document_in_top_3(retriever, query, expected_doc):
    hits = retriever.retrieve(query, 3)
    assert expected_doc in [h.doc_id for h in hits]


def test_results_are_ranked_and_scored(retriever):
    hits = retriever.retrieve("VPN connection timed out", 4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(hits[i].score >= hits[i + 1].score for i in range(3))