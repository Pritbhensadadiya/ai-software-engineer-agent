from pathlib import Path

import pytest

import search_codebase
from embeddings import DEFAULT_EMBEDDING_MODEL
from vector_store import VectorStore
from types import SimpleNamespace

import search_codebase

def test_main_displays_retrieved_code(monkeypatch, capsys, tmp_path):
    class FakeStore:
        def __len__(self):
            return 1

        def save(self, path, *, model):
            assert model == DEFAULT_EMBEDDING_MODEL
            Path(path).write_text("test index", encoding="utf-8")

    fake_store = FakeStore()
    index_path = tmp_path / "index.json"

    def fake_build_code_index(
        root,
        *,
        max_chunks,
        model,
        exclude_dirs,
        exclude_files,
    ):
        assert root == tmp_path.resolve()
        assert max_chunks == 3
        assert model == DEFAULT_EMBEDDING_MODEL
        assert exclude_dirs == set()
        assert exclude_files == set()
        return fake_store

    def fake_retrieve_code(question, store, *, top_k, model):
        assert question == "Where is path validation?"
        assert store is fake_store
        assert top_k == 2
        assert model == DEFAULT_EMBEDDING_MODEL

        return [
            {
                "file_path": "tools.py",
                "start_line": 10,
                "end_line": 20,
                "language": "Python",
                "score": 0.95,
                "content": "def validate_path(path): ...",
            }
        ]

    monkeypatch.setattr(
        search_codebase,
        "build_code_index",
        fake_build_code_index,
    )
    monkeypatch.setattr(
        search_codebase,
        "retrieve_code",
        fake_retrieve_code,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_codebase.py",
            "Where is path validation?",
            "--root",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--max-chunks",
            "3",
            "--top-k",
            "2",
        ],
    )

    assert search_codebase.main() == 0

    output = capsys.readouterr().out

    assert "Saved index:" in output
    assert "Indexed chunks: 1" in output
    assert "tools.py:10-20" in output
    assert "Similarity: 0.9500" in output
    assert "def validate_path(path): ..." in output


def test_main_handles_empty_index(monkeypatch, capsys, tmp_path):
    class EmptyStore:
        def __len__(self):
            return 0

        def save(self, path, *, model):
            Path(path).write_text("empty index", encoding="utf-8")

    def fake_build_code_index(
        root,
        *,
        max_chunks,
        model,
        exclude_dirs,
        exclude_files,
    ):
        assert exclude_dirs == set()
        assert exclude_files == set()
        return EmptyStore()

    monkeypatch.setattr(
        search_codebase,
        "build_code_index",
        fake_build_code_index,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_codebase.py",
            "Find a function",
            "--root",
            str(tmp_path),
            "--index-path",
            str(tmp_path / "index.json"),
        ],
    )

    assert search_codebase.main() == 0

    output = capsys.readouterr().out
    assert "No supported code chunks were found." in output


def test_main_loads_saved_index_without_rebuilding(monkeypatch, capsys, tmp_path):
    index_path = tmp_path / "index.json"

    store = VectorStore()
    store.add(
        [
            {
                "file_path": "example.py",
                "start_line": 1,
                "end_line": 1,
                "language": "Python",
                "content": "def example(): pass",
                "embedding": [1.0, 0.0],
            }
        ]
    )
    store.save(index_path, model=DEFAULT_EMBEDDING_MODEL)

    def unexpected_build(*args, **kwargs):
        pytest.fail("The existing index should be loaded, not rebuilt.")

    monkeypatch.setattr(
        search_codebase,
        "build_code_index",
        unexpected_build,
    )
    monkeypatch.setattr(
        search_codebase,
        "retrieve_code",
        lambda question, store, *, top_k, model: [],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_codebase.py",
            "Find an example",
            "--root",
            str(tmp_path),
            "--index-path",
            str(index_path),
        ],
    )

    assert search_codebase.main() == 0

    output = capsys.readouterr().out
    assert "Loading saved index:" in output
    assert "Indexed chunks: 1" in output
    assert "No matching code chunks found." in output


def test_main_rebuilds_existing_index(monkeypatch, capsys, tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text("old index", encoding="utf-8")

    class FakeStore:
        def __len__(self):
            return 0

        def save(self, path, *, model):
            Path(path).write_text("rebuilt index", encoding="utf-8")

    def fake_build_code_index(
        root,
        *,
        max_chunks,
        model,
        exclude_dirs,
        exclude_files,
    ):
        assert exclude_dirs == set()
        assert exclude_files == set()
        return FakeStore()

    monkeypatch.setattr(
        search_codebase,
        "build_code_index",
        fake_build_code_index,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_codebase.py",
            "Find a function",
            "--root",
            str(tmp_path),
            "--index-path",
            str(index_path),
            "--rebuild",
        ],
    )

    assert search_codebase.main() == 0

    output = capsys.readouterr().out
    assert "Indexing repository:" in output
    assert "Saved index:" in output
    assert index_path.read_text(encoding="utf-8") == "rebuilt index"


def test_main_passes_exclusions_to_index_builder(
    monkeypatch,
    capsys,
    tmp_path,
):
    class EmptyStore:
        def __len__(self):
            return 0

        def save(self, path, *, model):
            Path(path).write_text("empty index", encoding="utf-8")

    def fake_build_code_index(
        root,
        *,
        max_chunks,
        model,
        exclude_dirs,
        exclude_files,
    ):
        assert exclude_dirs == {"scripts", "scratch"}
        assert exclude_files == {
            "tests/test_tools.py",
            "src/generated.py",
        }
        return EmptyStore()

    monkeypatch.setattr(
        search_codebase,
        "build_code_index",
        fake_build_code_index,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_codebase.py",
            "Find a function",
            "--root",
            str(tmp_path),
            "--index-path",
            str(tmp_path / "index.json"),
            "--exclude-dir",
            "scripts",
            "--exclude-dir",
            "scratch",
            "--exclude-file",
            "tests/test_tools.py",
            "--exclude-file",
            "src/generated.py",
        ],
    )

    assert search_codebase.main() == 0

    output = capsys.readouterr().out
    assert "Excluded directories: scripts, scratch" in output
    assert "Excluded files: tests/test_tools.py, src/generated.py" in output


def test_fresh_index_reports_success(tmp_path, monkeypatch, capsys):
    manifest = {"app.py": {"size": 10, "mtime_ns": 100}}
    store = SimpleNamespace(
        metadata={
            "root": str(tmp_path.resolve()),
            "source_manifest": manifest,
            "exclude_dirs": [],
            "exclude_files": [],
        }
    )

    monkeypatch.setattr(
        search_codebase,
        "build_source_manifest",
        lambda *args, **kwargs: manifest,
    )

    search_codebase.check_index_freshness(store, tmp_path)

    assert "freshness check passed" in capsys.readouterr().out.lower()


def test_stale_index_reports_file_changes(tmp_path, monkeypatch, capsys):
    saved_manifest = {
        "changed.py": {"size": 10, "mtime_ns": 100},
        "removed.py": {"size": 10, "mtime_ns": 100},
    }
    current_manifest = {
        "changed.py": {"size": 20, "mtime_ns": 200},
        "added.py": {"size": 5, "mtime_ns": 300},
    }
    store = SimpleNamespace(
        metadata={
            "root": str(tmp_path.resolve()),
            "source_manifest": saved_manifest,
            "exclude_dirs": [],
            "exclude_files": [],
        }
    )

    monkeypatch.setattr(
        search_codebase,
        "build_source_manifest",
        lambda *args, **kwargs: current_manifest,
    )

    search_codebase.check_index_freshness(store, tmp_path)

    output = capsys.readouterr().out
    assert "stale" in output.lower()
    assert "added.py" in output
    assert "changed.py" in output
    assert "removed.py" in output
    assert "--rebuild" in output


def test_legacy_index_without_manifest_warns(tmp_path, capsys):
    store = SimpleNamespace(metadata={})

    search_codebase.check_index_freshness(store, tmp_path)

    output = capsys.readouterr().out
    assert "no source manifest" in output.lower()
    assert "--rebuild" in output


def test_different_repository_root_warns(tmp_path, capsys):
    store = SimpleNamespace(
        metadata={
            "root": str(tmp_path / "another-repo"),
            "source_manifest": {},
        }
    )

    search_codebase.check_index_freshness(store, tmp_path)

    output = capsys.readouterr().out
    assert "different repository root" in output.lower()
    assert "--rebuild" in output


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {"app.py": None},
        {"app.py": {"size": -1, "mtime_ns": 100}},
        {"app.py": {"size": 10, "mtime_ns": True}},
        {123: {"size": 10, "mtime_ns": 100}},
    ],
)
def test_invalid_source_manifest_warns_and_returns(
    tmp_path, capsys, monkeypatch, manifest
):
    store = SimpleNamespace(
        metadata={
            "root": str(tmp_path.resolve()),
            "source_manifest": manifest,
            "exclude_dirs": [],
            "exclude_files": [],
        }
    )

    def unexpected_manifest_build(*args, **kwargs):
        pytest.fail("Invalid metadata should be rejected before scanning.")

    monkeypatch.setattr(
        search_codebase,
        "build_source_manifest",
        unexpected_manifest_build,
    )

    search_codebase.check_index_freshness(store, tmp_path)

    output = capsys.readouterr().out
    if manifest is None:
        assert "no source manifest" in output.lower()
    else:
        assert "invalid source manifest" in output.lower()

        assert "--rebuild" in output
        assert "--rebuild" in output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exclude_dirs", "scratch"),
        ("exclude_dirs", [None]),
        ("exclude_files", [""]),
        ("exclude_files", [["nested.py"]]),
    ],
)
def test_invalid_exclusion_metadata_warns_and_returns(
    tmp_path, capsys, monkeypatch, field, value
):
    metadata = {
        "root": str(tmp_path.resolve()),
        "source_manifest": {},
        "exclude_dirs": [],
        "exclude_files": [],
    }
    metadata[field] = value
    store = SimpleNamespace(metadata=metadata)

    def unexpected_manifest_build(*args, **kwargs):
        pytest.fail("Invalid exclusion metadata should be rejected.")

    monkeypatch.setattr(
        search_codebase,
        "build_source_manifest",
        unexpected_manifest_build,
    )

    search_codebase.check_index_freshness(store, tmp_path)

    output = capsys.readouterr().out
    assert "invalid" in output.lower()
    assert "--rebuild" in output
