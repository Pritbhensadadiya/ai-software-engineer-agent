import main


class FakeStore:
    def __len__(self):
        return 42


def test_main_passes_workflow_dependencies(
    monkeypatch,
    capsys,
):
    fake_store = FakeStore()
    fake_client = object()

    calls = {}

    monkeypatch.setattr(
        main,
        "genai",
        type(
            "FakeGenAI",
            (),
            {"Client": staticmethod(lambda api_key: fake_client)},
        ),
    )

    monkeypatch.setattr(
        main,
        "load_or_build_code_index",
        lambda client: fake_store,
    )

    def fake_run_workflow(
        task,
        *,
        client,
        code_store,
        embedding_client,
    ):
        calls["task"] = task
        calls["client"] = client
        calls["code_store"] = code_store
        calls["embedding_client"] = embedding_client

        return type(
            "FakeWorkflowResult",
            (),
            {
                "status": "completed",
                "stages": [
                    "planner",
                    "developer",
                    "tester",
                    "reviewer",
                ],
                "output": "mock workflow answer",
                "error": None,
            },
        )()

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_run_workflow,
    )

    monkeypatch.setattr(
        main,
        "input",
        lambda _: "How does the repository analyzer work?",
        raising=False,
    )

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "fake-api-key",
    )

    main.main()

    assert calls == {
        "task": "How does the repository analyzer work?",
        "client": fake_client,
        "code_store": fake_store,
        "embedding_client": fake_client,
    }

    output = capsys.readouterr().out

    assert "Code index ready: 42 chunks." in output
    assert "Status: completed" in output
    assert "planner -> developer -> tester -> reviewer" in output
    assert "mock workflow answer" in output


def test_main_rejects_empty_task(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "",
    )

    main.main()

    captured = capsys.readouterr()

    assert "Please enter a task." in captured.out


def test_main_requires_api_key(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "Build a feature.",
    )

    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    main.main()

    captured = capsys.readouterr()

    assert "GEMINI_API_KEY is missing." in captured.out
