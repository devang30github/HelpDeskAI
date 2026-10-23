"""Deterministic pieces of the in-chat ticket flow: yes/no detection, category and
priority parsing, summaries and the next-step decision. No LLM, no database."""
import re

from app.core.constants import CATEGORY_LABELS, Category, Priority

YES = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "go ahead", "do it",
    "create it", "create ticket", "create a ticket", "create the ticket", "submit",
    "yes please", "yes confirm", "yes create ticket", "yes create a ticket", "yes create the ticket",
}
NO = {
    "no", "n", "nope", "nah", "cancel", "cancel it", "stop", "abort", "never mind", "nevermind",
    "not now", "no thanks", "no thank you", "dont", "do not", "dont create", "no cancel",
    "no dont create a ticket",
}

ASK_DESCRIPTION = (
    "Happy to raise a ticket. Please describe the problem in a sentence or two: "
    "what is happening, any error message, and your location or device."
)
NEED_DETAIL = "Could you add a little more detail (at least a short sentence) so the helpdesk team can act on it?"
ASK_CATEGORY = "Which category fits best? Pick one below or type it."
CATEGORY_REPROMPT = "I couldn't match that to a category. Please pick one of the options below."
CANCEL_REPLY = "Okay, I haven't created a ticket. Is there anything else I can help with?"
CONFIRM_QUICK = ["Yes, create the ticket", "Cancel"]


def normalize(text: str) -> str:
    text = text.lower().replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def is_yes(text: str) -> bool:
    return normalize(text) in YES


def is_no(text: str) -> bool:
    return normalize(text) in NO


_LABEL_NORM = {c: normalize(CATEGORY_LABELS[c]) for c in Category}
_VALUE_NORM = {c: normalize(c.value) for c in Category}
_HINTS = {
    Category.NETWORK_WIFI: ("wifi", "wi fi", "network", "vpn", "internet"),
    Category.HARDWARE: ("hardware", "laptop", "notebook", "monitor", "keyboard", "mouse", "peripheral", "headset", "dock"),
    Category.SOFTWARE: ("software", "application", "install", "licen"),
    Category.EMAIL_COLLAB: ("email", "e mail", "mail", "outlook", "calendar", "teams", "collaborat", "meeting"),
    Category.ACCOUNT_ACCESS: ("account", "password", "login", "log in", "sign in", "access", "permission", "mfa"),
    Category.PRINTING: ("print", "scan", "toner", "copier"),
    Category.FACILITIES: ("facilit", "maintenance", "light", "air condition", "leak", "desk", "chair", "temperature"),
}


def _exact_category(n: str) -> str | None:
    if n.isdigit() and 1 <= int(n) <= len(Category):
        return list(Category)[int(n) - 1].value
    for c in Category:
        if n in (_LABEL_NORM[c], _VALUE_NORM[c]):
            return c.value
    return None


def parse_category(text: str) -> str | None:
    n = normalize(text)
    exact = _exact_category(n)
    if exact:
        return exact
    for c in Category:
        if _LABEL_NORM[c] in n or _VALUE_NORM[c] in n:
            return c.value
    matches = [c for c, words in _HINTS.items() if any(re.search(rf"\b{re.escape(w)}", n) for w in words)]
    return matches[0].value if len(matches) == 1 else None  # ambiguous -> ask


def parse_priority_edit(text: str) -> str | None:
    n = normalize(text)
    found = [p.value for p in Priority if re.search(rf"\b{p.value}\b", n)]
    if len(found) != 1:
        return None
    if n == found[0] or re.search(r"\b(priority|make|set|change|mark|urgent)\b", n):
        return found[0]
    return None


def derive_title(description: str) -> str:
    first = re.split(r"(?<=[.!?])\s|\n", description.strip(), maxsplit=1)[0].strip()
    if len(first) > 80:
        first = first[:80].rsplit(" ", 1)[0] + "..."
    return first if len(first) >= 5 else "Support request"


def summary_text(draft: dict) -> str:
    label = CATEGORY_LABELS[Category(draft["category"])]
    return (
        "Here's the ticket I'm about to create:\n\n"
        f"Title: {draft['title']}\n"
        f"Category: {label}\n"
        f"Priority: {draft['priority'].title()}\n"
        f"Description: {draft['description']}\n\n"
        'Reply "yes" to create it, "cancel" to stop, or tell me what to change '
        '(for example "make it high priority").'
    )


def advance(draft: dict) -> tuple[str, str, list[str]]:
    """Decide the next step from the draft. Returns (mode, reply, quick_replies).
    Fills the title and a default priority once the required fields are present."""
    if not draft.get("description"):
        return "collecting", ASK_DESCRIPTION, []
    if not draft.get("category"):
        return "collecting", ASK_CATEGORY, [CATEGORY_LABELS[c] for c in Category]
    if not draft.get("title"):
        draft["title"] = derive_title(draft["description"])
    if not draft.get("priority"):
        draft["priority"] = Priority.MEDIUM.value
    return "confirming", summary_text(draft), list(CONFIRM_QUICK)


def apply_confirming_edit(text: str, draft: dict) -> dict:
    """Handle a non-yes/no message while the summary is showing."""
    priority = parse_priority_edit(text)
    if priority:
        draft["priority"] = priority
        return draft
    n = normalize(text)
    if re.search(r"\bcategor", n):
        category = parse_category(text)
        if category:
            draft["category"] = category
            return draft
    else:
        exact = _exact_category(n)
        if exact:
            draft["category"] = exact
            return draft
    # Anything else is extra detail for the description (never silently changes the category).
    draft["description"] = (draft["description"] + "\n\nAdditional details: " + text.strip())[:5000]
    return draft