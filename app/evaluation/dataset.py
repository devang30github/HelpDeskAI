"""Evaluation dataset: schema, loading and validation against the knowledge base."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.constants import Category
from app.rag.ingestion import load_documents

VALID_CATEGORIES = {c.value for c in Category} | {"general"}
QUESTION_TYPES = ("direct", "multi_chunk", "paraphrase", "ambiguous", "error_code", "confusable", "unanswerable")


class DatasetError(Exception):
    pass


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error, not silently ignored

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    split: Literal["dev", "test"]
    category: str
    type: Literal["direct", "multi_chunk", "paraphrase", "ambiguous", "error_code", "confusable", "unanswerable"]
    question: str = Field(min_length=8, max_length=500)
    answerable: bool
    reference_answer: str | None = None
    reference_doc_ids: list[str] = Field(default_factory=list)
    reference_contexts: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    origin: Literal["human_written", "synthetic_llm_drafted"]
    validated: bool = False
    validated_by: str | None = None
    smoke: bool = False
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("category")
    @classmethod
    def known_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"unknown category '{v}'")
        return v

    @model_validator(mode="after")
    def consistent(self):
        if (self.type == "unanswerable") == self.answerable:
            raise ValueError("answerable must be false exactly when type is 'unanswerable'")
        if self.smoke and self.split != "dev":
            raise ValueError("smoke questions must be in the dev split")
        if self.answerable:
            if not (self.reference_answer or "").strip():
                raise ValueError("answerable questions need a reference_answer")
            if not self.reference_doc_ids:
                raise ValueError("answerable questions need reference_doc_ids")
            if not self.reference_contexts:
                raise ValueError("answerable questions need reference_contexts")
        else:
            if not (self.abstention_reason or "").strip():
                raise ValueError("unanswerable questions need an abstention_reason")
            if self.reference_answer or self.reference_doc_ids or self.reference_contexts:
                raise ValueError("unanswerable questions must not have a reference answer, documents or contexts")
        return self


def load_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    errors: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = EvalItem.model_validate_json(line)
        except ValidationError as e:
            detail = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}" for err in e.errors())
            errors.append(f"line {number}: {detail}")
            continue
        if item.id in seen:
            errors.append(f"line {number}: duplicate id '{item.id}'")
            continue
        seen.add(item.id)
        items.append(item)

    groups: dict[str, set[str]] = {}
    for item in items:
        if item.group:
            groups.setdefault(item.group, set()).add(item.split)
    for group, splits in sorted(groups.items()):
        if len(splits) > 1:
            errors.append(f"group '{group}' spans several splits {sorted(splits)} (this leaks test data into dev)")

    if errors:
        raise DatasetError("\n".join(errors))
    return items


def select_items(items: list[EvalItem], split: str) -> list[EvalItem]:
    if split == "full":
        chosen = list(items)
    elif split == "smoke":
        chosen = [i for i in items if i.smoke]
    elif split in ("dev", "test"):
        chosen = [i for i in items if i.split == split]
    else:
        raise DatasetError(f"unknown split '{split}'")
    if not chosen:
        raise DatasetError(f"no questions selected for split '{split}'")
    return chosen


def label_hash(items: list[EvalItem]) -> str:
    """Hash of the label content only (not the validated flag), so two runs can prove
    they were scored against identical questions and references."""
    fields = ("id", "split", "category", "type", "question", "answerable", "reference_answer",
              "reference_doc_ids", "reference_contexts", "abstention_reason")
    canonical = [{f: getattr(i, f) for f in fields} for i in items]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta_path(dataset_path: Path) -> Path:
    return Path(dataset_path).with_name("dataset_meta.json")


def load_meta(dataset_path: Path) -> dict:
    p = meta_path(dataset_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _norm(text: str) -> str:
    return " ".join(text.split())


def validate_against_kb(items: list[EvalItem], kb_dir: Path) -> list[str]:
    """Check the labels against the real documents. Returns a list of problems."""
    docs, _ = load_documents(kb_dir)
    texts = {d.doc_id: _norm(d.text) for d in docs}
    problems: list[str] = []
    for item in items:
        for doc_id in item.reference_doc_ids:
            if doc_id not in texts:
                problems.append(f"{item.id}: reference_doc_id '{doc_id}' does not exist in the knowledge base")
        for context in item.reference_contexts:
            quote = _norm(context)
            if not any(quote in texts.get(d, "") for d in item.reference_doc_ids):
                problems.append(f"{item.id}: quote not found verbatim in the referenced documents: '{quote[:70]}...'")
    return problems


def check_kb_pin(meta: dict, indexed_files: dict[str, str]) -> list[str]:
    pinned = meta.get("kb_files") or {}
    if not pinned:
        return ["dataset_meta.json has no kb_files pin (run: python -m scripts.validate_dataset --pin-kb)"]
    problems = []
    for name in sorted(set(pinned) | set(indexed_files)):
        if pinned.get(name) != indexed_files.get(name):
            problems.append(f"{name}: pinned={str(pinned.get(name))[:12]} indexed={str(indexed_files.get(name))[:12]}")
    return problems


def coverage(items: list[EvalItem]) -> dict:
    answerable = [i for i in items if i.answerable]
    return {
        "total": len(items),
        "answerable": len(answerable),
        "unanswerable": len(items) - len(answerable),
        "by_split": dict(Counter(i.split for i in items)),
        "by_type": dict(Counter(i.type for i in items)),
        "answerable_by_category": dict(Counter(i.category for i in answerable)),
        "smoke": sum(i.smoke for i in items),
        "validated": sum(i.validated for i in items),
        "origins": dict(Counter(i.origin for i in items)),
    }


def coverage_warnings(items: list[EvalItem]) -> list[str]:
    cov, warnings = coverage(items), []
    if cov["total"] < 30:
        warnings.append(f"only {cov['total']} questions (the plan calls for about 30 to 50)")
    for category in Category:
        if cov["answerable_by_category"].get(category.value, 0) < 2:
            warnings.append(f"fewer than 2 answerable questions for category '{category.value}'")
    for qtype in QUESTION_TYPES:
        if cov["by_type"].get(qtype, 0) < 3:
            warnings.append(f"fewer than 3 questions of type '{qtype}'")
    if cov["unanswerable"] < 8:
        warnings.append("fewer than 8 unanswerable questions")
    for split in ("dev", "test"):
        if cov["by_split"].get(split, 0) == 0:
            warnings.append(f"no questions in the {split} split")
    if cov["smoke"] < 5:
        warnings.append("fewer than 5 smoke questions")
    return warnings