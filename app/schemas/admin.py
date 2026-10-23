from pydantic import BaseModel


class StatsOut(BaseModel):
    total: int
    by_status: dict[str, int]
    by_priority: dict[str, int]
    by_category: dict[str, int]
    overdue: int
    unassigned_open: int