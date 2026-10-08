from pathlib import Path

from code_index_manager import load_or_build_code_index


class FakeStore:
    def __init__(self):
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(Path(path))


def test_builds_and_saves_when_index_does_not_exist(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()

    index_path = tmp_path / "index.json"

    fake_store = FakeStore()
    calls = {}

    def fake_build_code_index(
        received_root,
        *,
        client=None,
        model=None,
        max_chunks=None,
        exclude_dirs=None,
        exclude_files=None,
    ):
        calls["build"] = {
            "root": received_root,
            "client": client,
            "model": model,
            "max_chunks": max_chunks,
        }
        return fake_store

    monkeypatch.setattr(
        "code_index_manager.build_code_index",
        fake_build_code_index,
    )

    result = load_or_build_code_index(
        root,
        index_path=index_path,
        client="fake-client",
        model="fake-model",
        max_chunks=100,
    )

    assert result is fake_store

    assert calls["build"] == {
        "root": root.resolve(),
        "client": "fake-client",
        "model": "fake-model",
        "max_chunks": 100,
    }

    assert fake_store.saved_paths == [index_path.resolve()]


def test_loads_and_updates_existing_index(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "repo"
    root.mkdir()

    index_path = tmp_path / "index.json"
    index_path.write_text(
        "existing",
        encoding="utf-8",
    )

    fake_store = FakeStore()
    calls = {}

    def fake_load(path):
        calls["load_path"] = Path(path)
        return fake_store

    def fake_update_code_index(
        store,
        received_root,
        *,
        client=None,
        model=None,
        max_chunks=None,
        exclude_dirs=None,
        exclude_files=None,
    ):
        calls["update"] = {
            "store": store,
            "root": received_root,
            "client": client,
            "model": model,
            "max_chunks": max_chunks,
        }
        return store

    monkeypatch.setattr(
        "code_index_manager.VectorStore.load",
        fake_load,
    )

    monkeypatch.setattr(
        "code_index_manager.update_code_index",
        fake_update_code_index,
    )

    result = load_or_build_code_index(
        root,
        index_path=index_path,
        client="fake-client",
        model="fake-model",
        max_chunks=100,
    )

    assert result is fake_store

    assert calls["load_path"] == index_path.resolve()

    assert calls["update"] == {
        "store": fake_store,
        "root": root.resolve(),
        "client": "fake-client",
        "model": "fake-model",
        "max_chunks": 100,
    }

    # An existing index is refreshed and persisted.
    assert fake_store.saved_paths == [index_path.resolve()]
