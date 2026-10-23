from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_admin
from app.core.constants import CATEGORY_LABELS, TEAM_MEMBERS, Category, Priority, Status
from app.models.user import User
from app.schemas.ticket import TicketCreate, TicketCreateIn, TicketOut, TicketUpdate
from app.tickets import service

router = APIRouter(prefix="/api", tags=["tickets"])


@router.get("/meta")
def meta() -> dict:
    """Constants the frontends need, so values are never duplicated by hand."""
    return {
        "categories": [{"value": c.value, "label": CATEGORY_LABELS[c]} for c in Category],
        "priorities": [p.value for p in Priority],
        "statuses": [s.value for s in Status],
        "team_members": TEAM_MEMBERS,
    }


@router.post("/tickets", response_model=TicketOut, status_code=201)
def create_ticket(
    payload: TicketCreateIn,
    response: Response,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = TicketCreate(
        **payload.model_dump(exclude={"idempotency_key"}),
        requester=user.username,
        # Scope the key to the user so two users can never collide on the same key.
        idempotency_key=(
            f"{user.username}:{payload.idempotency_key}" if payload.idempotency_key else None
        ),
    )
    ticket, created = service.create_ticket(db, data)
    if not created:
        response.status_code = 200  # duplicate request: return the existing ticket
    return ticket


@router.get("/tickets", response_model=list[TicketOut], dependencies=[Depends(require_admin)])
def list_tickets(
    category: Category | None = None,
    priority: Priority | None = None,
    status: Status | None = None,
    assignee: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    return service.list_tickets(
        db,
        category=category.value if category else None,
        priority=priority.value if priority else None,
        status=status.value if status else None,
        assignee=assignee,
        q=q,
    )


@router.get(
    "/tickets/{ticket_id}", response_model=TicketOut, dependencies=[Depends(require_admin)]
)
def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    try:
        return service.get_ticket(db, ticket_id)
    except service.TicketNotFound:
        raise HTTPException(404, "Ticket not found")


@router.patch(
    "/tickets/{ticket_id}", response_model=TicketOut, dependencies=[Depends(require_admin)]
)
def update_ticket(ticket_id: str, payload: TicketUpdate, db: Session = Depends(get_db)):
    try:
        return service.update_ticket(db, ticket_id, payload)
    except service.TicketNotFound:
        raise HTTPException(404, "Ticket not found")
    except service.InvalidTransition as e:
        raise HTTPException(409, str(e))
    except service.TicketRuleViolation as e:
        raise HTTPException(422, str(e))