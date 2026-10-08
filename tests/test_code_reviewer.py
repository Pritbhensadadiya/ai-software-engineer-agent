import pytest

from code_reviewer import review_code


@pytest.mark.parametrize(
    ("code", "expected_status", "expected_severity"),
    [
        (
            "def example():\n    return None\n",
            "changes_requested",
            "warning",
        ),
        (
            "def example():\n    raise Exception('debug')\n",
            "changes_requested",
            "error",
        ),
        (
            "def example(value):\n    return value\n",
            "approved",
            None,
        ),
    ],
)
def test_code_reviewer_classifies_code(
    code,
    expected_status,
    expected_severity,
):
    review = review_code(
        {
            "file": "example.py",
            "code": code,
        }
    )

    assert review["status"] == expected_status

    if expected_severity:
        assert review["findings"][0]["severity"] == expected_severity
    else:
        assert review["findings"] == []
def test_code_reviewer_reports_changed_behavior():
    review = review_code(
        {
            "file": "example.py",
            "code": ("def calculate(value):\n    return value * 0\n"),
        }
    )

    assert review["status"] == "changes_requested"
    assert review["findings"]


def test_code_reviewer_preserves_approved_status_for_clean_code():
    review = review_code(
        {
            "file": "example.py",
            "code": ("def calculate(value):\n    return value * 2\n"),
        }
    )

    assert review["status"] == "approved"
    assert review["findings"] == []