from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    source_name: str
    text: str
    sha256: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_name: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)
    vector_id: int = -1  # position/ID of this chunk's vector in the FAISS index


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_name: str
    text: str
    score: float
    rank: int  # 1-based
    metadata: dict = field(default_factory=dict)