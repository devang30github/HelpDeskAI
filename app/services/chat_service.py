"""Chat orchestration: conversation storage, intent routing, RAG answers and the
ticket flow. The LLM suggests; this class (the backend) decides and validates."""
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.timeutils import utcnow
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.ticket import TicketCreateIn
from app.services import ticket_flow as flow
from app.services.intent import classify_intent
from app.services.ticket_drafting import suggest_draft
from app.tickets import service as tickets

logger = logging.getLogger(__name__)

GREETING_REPLY = (
    "Hi! I'm HelpDeskAI. I can help with IT support and facilities questions, and I can raise a "
    "support ticket if I can't solve something. What do you need help with?"
)
THANKS_REPLY = "You're welcome! Let me know if anything else comes up."
NEUTRAL_REPLY = "Is there something specific I can help you with?"
OUT_OF_SCOPE_REPLY = (
    "That's outside what I can help with. I only handle IT support and facilities questions "
    "(for example Wi-Fi, laptops, accounts, printing or building maintenance). "
    "For other topics, please contact the relevant department."
)
OFFER = "I can create a support ticket for the helpdesk team if you'd like. Shall I?"
OFFER_QUICK = ["Yes, create a ticket", "No, thanks"]
DECLINE_REPLY = "No problem. Let me know if there's anything else I can help with."
SHORT_FAIL = "I still couldn't find this in the knowledge base."


class ConversationNotFound(Exception):
    pass


@dataclass
class ChatReply:
    conversation_id: str
    reply: str
    intent: str
    mode: str
    answerable: bool | None = None
    sources: list[dict] = field(default_factory=list)
    ticket: dict | None = None
    quick_replies: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def _new_state() -> dict:
    return {"mode": "idle", "draft": {}, "ticket_seq": 0, "unanswered": 0}


class ChatService:
    def __init__(self, db: Session, pipeline_factory, llm_factory):
        self.db = db
        self.pipeline_factory = pipeline_factory
        self.llm_factory = llm_factory
        self.settings = get_settings()

    # ---------- persistence helpers ----------
    def _load(self, user: User, conversation_id: str | None) -> Conversation:
        if conversation_id is None:
            conv = Conversation(
                id=str(uuid.uuid4()), username=user.username, state_json=json.dumps(_new_state())
            )
            self.db.add(conv)  # persisted only when the turn commits
            return conv
        conv = self.db.get(Conversation, conversation_id)
        if conv is None or conv.username != user.username:
            raise ConversationNotFound(conversation_id)
        return conv

    @staticmethod
    def _state(conv: Conversation) -> dict:
        return {**_new_state(), **json.loads(conv.state_json or "{}")}

    def _messages(self, conv: Conversation) -> list[Message]:
        return list(
            self.db.scalars(
                select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)
            )
        )

    def _qa_history(self, conv: Conversation) -> list[dict]:
        """Recent question/answer turns only (ticket-flow chatter is excluded)."""
        qa = []
        for m in self._messages(conv):
            meta = json.loads(m.meta_json or "{}")
            if meta.get("kind") == "qa":
                qa.append({"role": m.role, "content": meta.get("history_text") or m.content})
        qa = qa[-self.settings.history_messages :]
        while qa and qa[0]["role"] != "user":
            qa = qa[1:]
        return qa

    def _transcript(self, conv: Conversation, extra_user: str | None = None) -> str:
        lines = [
            f"{'Employee' if m.role == 'user' else 'HelpDeskAI'}: {m.content[:500]}"
            for m in self._messages(conv)[-14:]
        ]
        if extra_user:
            lines.append(f"Employee: {extra_user[:500]}")
        return "\n".join(lines)

    def _commit_turn(
        self, conv, state, user_text, kind, reply, intent, *,
        answerable=None, sources=None, ticket=None, quick_replies=None, debug=None, history_text=None,
    ) -> ChatReply:
        sources, quick_replies = sources or [], quick_replies or []
        self.db.add(
            Message(conversation_id=conv.id, role="user", content=user_text,
                    meta_json=json.dumps({"kind": kind}))
        )
        meta = {"kind": kind, "intent": intent, "answerable": answerable, "sources": sources,
                "ticket": ticket, "quick_replies": quick_replies}
        if history_text:
            meta["history_text"] = history_text
        self.db.add(
            Message(conversation_id=conv.id, role="assistant", content=reply, meta_json=json.dumps(meta))
        )
        conv.state_json = json.dumps(state)
        conv.updated_at = utcnow()
        self.db.commit()
        return ChatReply(
            conversation_id=conv.id, reply=reply, intent=intent, mode=state["mode"],
            answerable=answerable, sources=sources, ticket=ticket,
            quick_replies=quick_replies, debug=debug or {},
        )

    # ---------- public API ----------
    def handle(self, user: User, text: str, conversation_id: str | None = None) -> ChatReply:
        text = text.strip()
        conv = self._load(user, conversation_id)
        state = self._state(conv)

        if state["mode"] in ("collecting", "confirming"):
            return self._ticket_step(user, conv, state, text)

        if state["mode"] == "ticket_offered":
            if flow.is_yes(text):
                return self._start_ticket(conv, state, text)
            if flow.is_no(text):
                state["mode"] = "idle"
                return self._commit_turn(conv, state, text, "ticket", DECLINE_REPLY, "chitchat")
            state["mode"] = "idle"  # offer ignored: treat as a normal message

        recent = [m for m in self._qa_history(conv)]
        intent, source = classify_intent(text, recent, self.llm_factory)
        debug = {"intent_source": source}

        if intent == "ticket_request":
            return self._start_ticket(conv, state, text, debug)
        if intent == "chitchat":
            if flow.is_yes(text) or flow.is_no(text):
                reply = NEUTRAL_REPLY
            elif re.search(r"\b(thanks|thank you|thx|bye|goodbye)\b", text, re.IGNORECASE):
                reply = THANKS_REPLY
            else:
                reply = GREETING_REPLY
            return self._commit_turn(conv, state, text, "chat", reply, intent, debug=debug)
        if intent == "out_of_scope":
            return self._commit_turn(conv, state, text, "chat", OUT_OF_SCOPE_REPLY, intent, debug=debug)
        return self._answer_question(conv, state, text, debug)

    def get_conversation(self, user: User, conversation_id: str) -> tuple[Conversation, list[Message]]:
        conv = self._load(user, conversation_id)
        return conv, self._messages(conv)

    # ---------- question answering ----------
    def _answer_question(self, conv, state, text, debug) -> ChatReply:
        history = self._qa_history(conv)
        result = self.pipeline_factory().answer(text, history=history)
        sources = [asdict(s) for s in result.sources]
        debug = {
            **debug,
            "timings": result.timings,
            "usage": result.usage,
            "pipeline": {k: result.config.get(k) for k in ("pipeline", "prompt_version", "retriever", "top_k")},
        }
        if result.answerable:
            state["unanswered"] = 0
            reply, quick = result.answer, []
        else:
            state["unanswered"] = int(state.get("unanswered", 0)) + 1
            explanation = result.answer if state["unanswered"] < 2 else SHORT_FAIL
            reply, quick = f"{explanation}\n\n{OFFER}", list(OFFER_QUICK)
            state["mode"] = "ticket_offered"
        return self._commit_turn(
            conv, state, text, "qa", reply, "kb_question", answerable=result.answerable,
            sources=sources, quick_replies=quick, debug=debug, history_text=result.answer,
        )

    # ---------- ticket flow ----------
    def _start_ticket(self, conv, state, text, debug=None) -> ChatReply:
        draft = suggest_draft(self._transcript(conv, extra_user=text), self.llm_factory).to_dict()
        mode, reply, quick = flow.advance(draft)
        state.update(mode=mode, draft=draft)
        return self._commit_turn(
            conv, state, text, "ticket", reply, "ticket_request", quick_replies=quick, debug=debug
        )

    def _ticket_step(self, user, conv, state, text) -> ChatReply:
        draft = dict(state.get("draft") or {})

        if flow.is_no(text):
            state.update(mode="idle", draft={})
            return self._commit_turn(conv, state, text, "ticket", flow.CANCEL_REPLY, "ticket_request")

        if state["mode"] == "confirming":
            if flow.is_yes(text):
                return self._create_ticket(user, conv, state, draft, text)
            draft = flow.apply_confirming_edit(text, draft)
        elif not draft.get("description"):
            if len(text) < 10:
                return self._commit_turn(
                    conv, state, text, "ticket", flow.NEED_DETAIL, "ticket_request"
                )
            draft["description"] = text[:5000]
            suggestion = suggest_draft(self._transcript(conv, extra_user=text), self.llm_factory).to_dict()
            for key in ("title", "category", "priority"):
                draft[key] = draft.get(key) or suggestion.get(key)
        elif not draft.get("category"):
            category = flow.parse_category(text)
            if category is None:
                return self._commit_turn(
                    conv, state, text, "ticket", flow.CATEGORY_REPROMPT, "ticket_request",
                    quick_replies=flow.advance(dict(draft))[2],
                )
            draft["category"] = category

        mode, reply, quick = flow.advance(draft)
        state.update(mode=mode, draft=draft)
        return self._commit_turn(
            conv, state, text, "ticket", reply, "ticket_request", quick_replies=quick
        )

    def _create_ticket(self, user, conv, state, draft, text) -> ChatReply:
        seq = int(state.get("ticket_seq", 0)) + 1
        try:
            payload = TicketCreateIn(
                title=draft["title"],
                description=draft["description"],
                category=draft["category"],
                priority=draft["priority"],
                conversation_id=conv.id,
                transcript_summary=self._transcript(conv, extra_user=text)[:8000],
                idempotency_key=f"{conv.id}:{seq}",  # a repeated 'yes' cannot create a second ticket
            )
        except (ValidationError, KeyError):
            logger.warning("ticket draft failed validation; restarting description step")
            state.update(mode="collecting", draft={})
            return self._commit_turn(
                conv, state, text, "ticket",
                "I couldn't validate those ticket details. " + flow.ASK_DESCRIPTION, "ticket_request",
            )

        ticket, _created = tickets.create_ticket_for_user(self.db, user.username, payload)
        state.update(mode="idle", draft={}, ticket_seq=seq, unanswered=0)
        ref = {
            "ticket_id": ticket.ticket_id, "title": ticket.title, "category": ticket.category,
            "priority": ticket.priority, "status": ticket.status,
        }
        reply = (
            f"Done. Your ticket {ticket.ticket_id} has been created with {ticket.priority} priority. "
            "The helpdesk team will follow up. Is there anything else I can help with?"
        )
        return self._commit_turn(conv, state, text, "ticket", reply, "ticket_request", ticket=ref)