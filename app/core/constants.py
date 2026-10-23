"""Single source of truth for categories, priorities, statuses and lifecycle rules.

Backend, database, tests and (later) frontend all use these exact values.
"""
from enum import Enum


class Category(str, Enum):
    NETWORK_WIFI = "network_wifi"
    HARDWARE = "hardware_peripherals"
    SOFTWARE = "software_applications"
    EMAIL_COLLAB = "email_collaboration"
    ACCOUNT_ACCESS = "account_access"
    PRINTING = "printing_office_equipment"
    FACILITIES = "facilities_maintenance"


CATEGORY_LABELS = {
    Category.NETWORK_WIFI: "Network and Wi-Fi",
    Category.HARDWARE: "Hardware and peripherals",
    Category.SOFTWARE: "Software and applications",
    Category.EMAIL_COLLAB: "Email and collaboration tools",
    Category.ACCOUNT_ACCESS: "Account access and permissions",
    Category.PRINTING: "Printing and office equipment",
    Category.FACILITIES: "Facilities and maintenance",
}


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


# Allowed lifecycle transitions: current status -> permitted next statuses
ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.OPEN: {Status.IN_PROGRESS, Status.CLOSED},
    Status.IN_PROGRESS: {Status.RESOLVED, Status.OPEN},
    Status.RESOLVED: {Status.CLOSED, Status.IN_PROGRESS},
    Status.CLOSED: set(),
}

# Hours until a ticket is considered overdue, by priority (used for due_at)
SLA_HOURS: dict[Priority, int] = {
    Priority.CRITICAL: 4,
    Priority.HIGH: 8,
    Priority.MEDIUM: 24,
    Priority.LOW: 72,
}

# SAMPLE helpdesk team (fictional names for development)
TEAM_MEMBERS: list[str] = ["Asha Verma", "Rohan Mehta", "Priya Nair", "Kabir Shah"]