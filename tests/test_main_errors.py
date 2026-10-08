import main


def test_main_handles_gemini_quota_error(monkeypatch, capsys):
    class FakeClient:
        pass

    def fake_input(prompt):
        return "Where is incremental code indexing implemented?"

    def fake_client(api_key):
        return FakeClient()

    def fake_load_or_build_code_index(**kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    monkeypatch.setattr(main, "input", fake_input, raising=False)
    monkeypatch.setattr(main.genai, "Client", fake_client)
    monkeypatch.setattr(
        main,
        "load_or_build_code_index",
        fake_load_or_build_code_index,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    main.main()

    captured = capsys.readouterr()

    assert "Gemini API quota is exhausted." in captured.out
    assert "required Gemini operation could not complete" in captured.out
    assert "check your Gemini API quota or billing" in captured.out
