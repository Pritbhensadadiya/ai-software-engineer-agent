
import pytest

from code_chunker import chunk_files, chunk_repository, chunk_text

def test_chunk_text_preserves_content_and_line_numbers():
    content = "\n".join(f"line {number}" for number in range(1, 11))

    chunks = chunk_text(
        content,
        file_path="src/example.py",
        language="Python",
        chunk_size=4,
        overlap=1,
    )

    assert [chunk["start_line"] for chunk in chunks] == [1, 4, 7]
    assert [chunk["end_line"] for chunk in chunks] == [4, 7, 10]
    assert chunks[0]["content"] == "line 1\nline 2\nline 3\nline 4"
    assert chunks[1]["content"] == "line 4\nline 5\nline 6\nline 7"
    assert chunks[0]["file_path"] == "src/example.py"
    assert chunks[0]["language"] == "Python"


def test_chunk_text_returns_empty_list_for_empty_content():
    assert (
        chunk_text(
            "",
            file_path="empty.py",
            language="Python",
        )
        == []
    )


def test_chunk_text_handles_content_without_trailing_newline():
    chunks = chunk_text(
        "first\nsecond\nthird",
        file_path="example.py",
        language="Python",
        chunk_size=2,
        overlap=0,
    )

    assert chunks[0]["content"] == "first\nsecond"
    assert chunks[1]["content"] == "third"
    assert chunks[1]["start_line"] == 3
    assert chunks[1]["end_line"] == 3


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_chunk_text_rejects_invalid_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text(
            "some code",
            file_path="example.py",
            language="Python",
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_repository_excludes_additional_directory(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scratch").mkdir()

    (tmp_path / "src" / "app.py").write_text(
        "def app():\n    return True",
        encoding="utf-8",
    )
    (tmp_path / "scratch" / "helper.py").write_text(
        "def helper():\n    return True",
        encoding="utf-8",
    )

    chunks = chunk_repository(
        root=tmp_path,
        exclude_dirs={"scratch"},
    )

    indexed_paths = {chunk["file_path"] for chunk in chunks}

    assert "src/app.py" in indexed_paths
    assert not any("scratch" in path.split("/") for path in indexed_paths)


def test_chunk_repository_excludes_specific_file(tmp_path):
    (tmp_path / "src").mkdir()

    (tmp_path / "src" / "app.py").write_text(
        "def app():\n    return True",
        encoding="utf-8",
    )
    (tmp_path / "src" / "helper.py").write_text(
        "def helper():\n    return True",
        encoding="utf-8",
    )

    chunks = chunk_repository(
        root=tmp_path,
        exclude_files={"src/helper.py"},
    )

    indexed_paths = {chunk["file_path"] for chunk in chunks}

    assert "src/app.py" in indexed_paths
    assert "src/helper.py" not in indexed_paths


def test_chunk_repository_preserves_default_ignored_directories(tmp_path):
    ignored_dir = tmp_path / ".venv"
    ignored_dir.mkdir()

    (ignored_dir / "ignored.py").write_text(
        "def ignored():\n    return True",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "def app():\n    return True",
        encoding="utf-8",
    )

    chunks = chunk_repository(root=tmp_path)
    indexed_paths = {chunk["file_path"] for chunk in chunks}

    assert "app.py" in indexed_paths
    assert not any(".venv" in path.split("/") for path in indexed_paths)

def test_chunk_files_chunks_only_selected_files(tmp_path):
    (tmp_path / "app.py").write_text(
        "def app():\n    return True",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "def helper():\n    return False",
        encoding="utf-8",
    )

    chunks = chunk_files(
        {"app.py"},
        root=tmp_path,
    )

    assert chunks
    assert {chunk["file_path"] for chunk in chunks} == {"app.py"}
    assert chunks[0]["language"] == "Python"


def test_chunk_files_returns_empty_for_missing_file(tmp_path):
    chunks = chunk_files(
        {"missing.py"},
        root=tmp_path,
    )

    assert chunks == []


def test_chunk_files_skips_unsupported_extension(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "Some notes",
        encoding="utf-8",
    )

    chunks = chunk_files(
        {"notes.txt"},
        root=tmp_path,
    )

    assert chunks == []


def test_chunk_files_skips_excluded_file(tmp_path):
    (tmp_path / "secret.py").write_text(
        "SECRET = True",
        encoding="utf-8",
    )

    chunks = chunk_files(
        {"secret.py"},
        root=tmp_path,
        exclude_files={"secret.py"},
    )

    assert chunks == []


def test_chunk_files_rejects_path_traversal(tmp_path):
    outside_file = tmp_path.parent / "outside.py"
    outside_file.write_text(
        "OUTSIDE = True",
        encoding="utf-8",
    )

    chunks = chunk_files(
        {"../outside.py"},
        root=tmp_path,
    )

    assert chunks == []


def test_chunk_files_rejects_invalid_chunk_parameters(tmp_path):
    with pytest.raises(ValueError):
        chunk_files(
            {"app.py"},
            root=tmp_path,
            chunk_size=0,
        )


def test_chunk_files_rejects_nonexistent_root(tmp_path):
    missing_root = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="existing directory"):
        chunk_files(
            {"app.py"},
            root=missing_root,
        )
