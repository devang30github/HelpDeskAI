"""Ticket-flow and category-classification evaluation. These run against a temporary
database: scripts/evaluate.py points DATABASE_URL at a temp file before app modules load."""
import json
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import BASE_DIR
from app.core.constants import CATEGORY_LABELS, Category, Priority, Status
from app.evaluation import metrics as M
from app.evaluation.dataset import DatasetError

SCENARIOS_PATH = BASE_DIR / "data" / "evaluation" / "ticket_scenarios_v1.jsonl"
CLASSIFICATION_PATH = BASE_DIR / "data" / "evaluation" / "classification_v1.jsonl"
CATEGORIES = [c.value for c in Category]
EVAL_USER = "eval-user"
REQUIRED_FIELDS = ("ticket_id", "title", "description", "category", "priority", "status", "requester",
                   "created_at", "updated_at", "conversation_id", "transcript_summary")


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    split: Literal["dev", "test"]
    expected_category: str
    validated: bool = False

    @field_validator("expected_category")
    @classmethod
    def known(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category '{v}'")
        return v


class Scenario(_Row):
    problem: str = Field(min_length=15, max_length=600)


class Example(_Row):
    text: str = Field(min_length=8, max_length=600)


def _load(path: Path, model):
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path} not found")
    rows, errors, seen = [], [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = model.model_validate_json(line)
        except ValidationError as e:
            detail = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}" for err in e.errors())
            errors.append(f"{path.name} line {number}: {detail}")
            continue
        if row.id in seen:
            errors.append(f"{path.name} line {number}: duplicate id '{row.id}'")
            continue
        seen.add(row.id)
        rows.append(row)
    if errors:
        raise DatasetError("\n".join(errors))
    return rows


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    return _load(path, Scenario)


def load_examples(path: Path = CLASSIFICATION_PATH) -> list[Example]:
    return _load(path, Example)


def select(rows: list, split: str) -> list:
    chosen = [r for r in rows if split == "full" or r.split == ("dev" if split == "smoke" else split)]
    if not chosen:
        raise DatasetError(f"no rows for split '{split}'")
    return chosen


def check_ticket(ticket, conversation_id: str) -> list[str]:
    problems = [f"missing:{f}" for f in REQUIRED_FIELDS if not getattr(ticket, f, None)]
    if ticket.category not in CATEGORIES:
        problems.append("invalid:category")
    if ticket.priority not in {p.value for p in Priority}:
        problems.append("invalid:priority")
    if ticket.status != Status.OPEN.value:
        problems.append("invalid:status")
    if ticket.conversation_id != conversation_id:
        problems.append("invalid:conversation_id")
    return problems


def _run_scenario(db, user, scenario: Scenario, llm_factory, max_turns: int) -> dict:
    from sqlalchemy import select as sql_select

    from app.models.ticket import Ticket
    from app.services.chat_service import ChatService

    def no_pipeline():
        raise RuntimeError("the ticket flow must not call the RAG pipeline")

    service = ChatService(db, pipeline_factory=no_pipeline, llm_factory=llm_factory)
    label = CATEGORY_LABELS[Category(scenario.expected_category)]
    result = {"id": scenario.id, "created": False, "turns": 0, "ticket_id": None, "complete": None,
              "missing_fields": [], "category": None, "expected_category": scenario.expected_category,
              "category_matches": None, "priority": None, "failure": None}
    reply, conversation_id, message = None, None, f"Please create a ticket. {scenario.problem}"
    try:
        while result["turns"] < max_turns:
            reply = service.handle(user, message, conversation_id)
            result["turns"] += 1
            conversation_id = reply.conversation_id
            if reply.ticket:
                break
            if reply.mode == "collecting":
                # The simulated employee answers truthfully: the problem, or the right category.
                message = label if reply.quick_replies else scenario.problem
            elif reply.mode == "confirming":
                message = "Yes, create the ticket"
            else:
                break
    except Exception as e:
        result["failure"] = f"{type(e).__name__}: {e}"
        return result

    if not (reply and reply.ticket):
        result["failure"] = f"no ticket after {result['turns']} turns (last mode: {reply.mode if reply else 'n/a'})"
        return result

    ticket = db.scalar(sql_select(Ticket).where(Ticket.ticket_id == reply.ticket["ticket_id"]))
    if ticket is None:
        result["failure"] = "ticket reported as created but not found in the database"
        return result
    problems = check_ticket(ticket, conversation_id)
    result.update(created=True, ticket_id=ticket.ticket_id, complete=not problems, missing_fields=problems,
                  category=ticket.category, category_matches=ticket.category == scenario.expected_category,
                  priority=ticket.priority)
    return result


def run_ticket_scenarios(scenarios, llm_factory=None, max_turns: int = 6, progress=None) -> list[dict]:
    from app.database.session import SessionLocal, init_db
    from app.rag.generation import get_llm
    from app.services.users import create_user, get_by_username

    init_db()
    llm_factory = llm_factory or get_llm
    results = []
    with SessionLocal() as db:
        if get_by_username(db, EVAL_USER) is None:
            create_user(db, EVAL_USER, "eval-password-123", "Eval User")
        user = get_by_username(db, EVAL_USER)
        for index, scenario in enumerate(scenarios, start=1):
            if progress:
                progress(f"  ticket scenario [{index}/{len(scenarios)}] {scenario.id}")
            results.append(_run_scenario(db, user, scenario, llm_factory, max_turns))
    return results


def ticket_summary(results: list[dict], n_unvalidated: int = 0) -> dict:
    created = [r for r in results if r["created"]]
    complete = [r for r in created if r["complete"]]
    matches = [r for r in created if r["category_matches"]]
    return {
        "n_scenarios": len(results),
        "n_unvalidated": n_unvalidated,
        "n_created": len(created),
        "creation_success_rate": len(created) / len(results) if results else None,
        "n_complete": len(complete),
        "completeness_rate": len(complete) / len(created) if created else None,
        "category_agreement": len(matches) / len(created) if created else None,
        "mean_turns": mean(r["turns"] for r in created) if created else None,
        "failures": [{"id": r["id"], "reason": r["failure"]} for r in results if not r["created"]],
    }


def run_classification(examples, llm_factory=None, progress=None) -> list[dict]:
    from app.rag.generation import LLMError, get_llm
    from app.services.llm_json import extract_json_object
    from app.services.ticket_drafting import DRAFT_SYSTEM_PROMPT, sanitize_draft

    llm_factory = llm_factory or get_llm
    results = []
    for index, example in enumerate(examples, start=1):
        if progress and index % 10 == 0:
            progress(f"  classification [{index}/{len(examples)}]")
        row = {"id": example.id, "text": example.text, "expected": example.expected_category,
               "predicted": None, "error": None}
        try:
            reply = llm_factory().complete([
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation:\nEmployee: {example.text}"},
            ])
            row["predicted"] = sanitize_draft(extract_json_object(reply.text)).category
        except LLMError as e:  # recorded and excluded from the metrics, never counted as a wrong answer
            row["error"] = str(e)
        results.append(row)
    return results


def classification_summary(results: list[dict], n_unvalidated: int = 0) -> dict:
    ok = [r for r in results if not r["error"]]
    report = M.classification_report(
        [r["expected"] for r in ok], [r["predicted"] or "none" for r in ok], CATEGORIES)
    report.update(
        n_unvalidated=n_unvalidated,
        n_errors=len(results) - len(ok),
        n_unclassified=sum(1 for r in ok if not r["predicted"]),
        errors=[{"id": r["id"], "error": r["error"]} for r in results if r["error"]],
    )
    return report