import pytest

from tools import list_files, read_file, run_tests, search_code


def test_list_files_lists_project_files(tmp_path):
    (tmp_path / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )

    files = list_files(root=tmp_path)

    assert "main.py" in files


def test_list_files_ignores_virtual_environment(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()

    (venv / "ignored.py").write_text(
        "secret",
        encoding="utf-8",
    )

    files = list_files(root=tmp_path)

    assert not any(".venv" in path for path in files)


def test_read_file_reads_text(tmp_path):
    (tmp_path / "app.py").write_text(
        "app = True",
        encoding="utf-8",
    )

    assert read_file("app.py", root=tmp_path) == "app = True"


def test_read_file_rejects_path_traversal(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text(
        "not allowed",
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="outside"):
            read_file("../outside.txt", root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_search_code_finds_matching_lines(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def login():\n    return 'token'\n",
        encoding="utf-8",
    )

    results = search_code("LOGIN", root=tmp_path)

    assert len(results) == 1
    assert results[0]["file"] == "auth.py"
    assert results[0]["line"] == 1


def test_search_code_rejects_empty_query(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        search_code("   ", root=tmp_path)


def test_run_tests_runs_pytest(tmp_path):
    (tmp_path / "test_example.py").write_text(
        """
def test_example():
    assert 2 + 2 == 4
""",
        encoding="utf-8",
    )

    result = run_tests(root=tmp_path)

    assert result["return_code"] == 0, (
        f"stdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
    )

    assert "1 passed" in result["stdout"]


def test_run_tests_reports_failure(tmp_path):
    (tmp_path / "test_example.py").write_text(
        """
def test_example():
    assert 2 + 2 == 5
""",
        encoding="utf-8",
    )

    result = run_tests(root=tmp_path)

    assert result["return_code"] != 0
    assert "1 failed" in result["stdout"]


def test_write_file_creates_a_new_file(tmp_path):
    from tools import write_file

    result = write_file(
        "test_example.py",
        "def test_example():\n    assert 2 + 2 == 4\n",
        root=tmp_path,
    )

    created_file = tmp_path / "test_example.py"

    assert created_file.exists()
    assert created_file.read_text(encoding="utf-8") == (
        "def test_example():\n    assert 2 + 2 == 4\n"
    )
    assert "Successfully wrote" in result
