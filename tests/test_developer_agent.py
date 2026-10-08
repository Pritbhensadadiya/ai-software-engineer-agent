from types import SimpleNamespace

import pytest

from developer_agent import run_developer


class FakeDeveloperClient:
    def __init__(self):
        self.interactions = self

    def create(self, **kwargs):
        return SimpleNamespace(
            id="developer-1",
            steps=[SimpleNamespace(type="message")],
            output_text=(
                "Implemented the requested change and verified the modified files."
            ),
        )


def test_developer_rejects_empty_plan():
    with pytest.raises(ValueError, match="must not be empty"):
        run_developer("", client=FakeDeveloperClient())


def test_developer_implements_plan():
    result = run_developer(
        "Add a logging feature.",
        client=FakeDeveloperClient(),
    )

    assert "Implemented the requested change" in result
    assert "verified" in result
