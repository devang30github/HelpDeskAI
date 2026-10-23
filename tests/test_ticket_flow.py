import pytest

from app.services import ticket_flow as flow


@pytest.mark.parametrize("text", ["yes", "Yes!", "  Yes, create the ticket ", "OK", "go ahead", "Create a ticket"])
def test_is_yes(text):
    assert flow.is_yes(text)


@pytest.mark.parametrize("text", ["yes but make it critical", "maybe", "no", "what about priority"])
def test_is_not_yes(text):
    assert not flow.is_yes(text)


@pytest.mark.parametrize("text", ["Cancel", "No, thanks", "don't", "Never mind"])
def test_is_no(text):
    assert flow.is_no(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Network and Wi-Fi", "network_wifi"),
        ("3", "software_applications"),
        ("email_collaboration", "email_collaboration"),
        ("my printer is jammed", "printing_office_equipment"),
        ("something unrelated", None),
        ("wifi and printer", None),  # ambiguous -> ask
    ],
)
def test_parse_category(text, expected):
    assert flow.parse_category(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("make it high priority", "high"),
        ("critical", "critical"),
        ("my laptop is critical for work", None),
        ("set priority to low and high", None),
    ],
)
def test_parse_priority_edit(text, expected):
    assert flow.parse_priority_edit(text) == expected


def test_derive_title():
    assert flow.derive_title("VPN times out. It started today.") == "VPN times out."
    assert flow.derive_title("ab") == "Support request"
    long = flow.derive_title("word " * 40)
    assert len(long) <= 84 and long.endswith("...")


def test_advance_walks_through_the_required_fields():
    mode, reply, quick = flow.advance({})
    assert mode == "collecting" and "describe" in reply and quick == []

    mode, _, quick = flow.advance({"description": "My monitor flickers constantly."})
    assert mode == "collecting" and len(quick) == 7

    draft = {"description": "My monitor flickers constantly.", "category": "hardware_peripherals"}
    mode, reply, quick = flow.advance(draft)
    assert mode == "confirming"
    assert draft["priority"] == "medium" and draft["title"] == "My monitor flickers constantly."
    assert "Hardware and peripherals" in reply and quick[0].startswith("Yes")


def test_confirming_edits():
    draft = {"title": "T title", "description": "Original text here.", "category": "network_wifi", "priority": "low"}
    assert flow.apply_confirming_edit("make it high priority", draft)["priority"] == "high"
    assert flow.apply_confirming_edit("change category to printing", draft)["category"] == "printing_office_equipment"
    flow.apply_confirming_edit("my printer is also broken", draft)
    assert draft["category"] == "printing_office_equipment"  # free text never changes the category
    assert "Additional details: my printer is also broken" in draft["description"]