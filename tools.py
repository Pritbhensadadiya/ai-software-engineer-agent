import difflib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
}

MAX_FILES = 200
MAX_FILE_BYTES = 200_000
MAX_MATCHES = 100


def _resolve_safe_path(path: str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path and reject paths outside the allowed project root."""
    root = root.resolve()
    candidate = (root / path).resolve()

    if not candidate.is_relative_to(root):
        raise ValueError("Path is outside the allowed project directory.")

    return candidate


def _list_files(
    directory: str = ".",
    root: Path = PROJECT_ROOT,
    *,
    max_files: int | None = MAX_FILES,
) -> list[str]:
    """List project files with an optional maximum file count."""
    base = _resolve_safe_path(directory, root)

    if not base.is_dir():
        raise ValueError("Directory does not exist.")

    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be greater than zero or None.")

    results = []

    for path in base.rglob("*"):
        relative = path.relative_to(root.resolve())

        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        if path.is_file():
            results.append(relative.as_posix())

            if max_files is not None and len(results) >= max_files:
                break

    return sorted(results)


def list_files(
    directory: str = ".",
    root: Path = PROJECT_ROOT,
) -> list[str]:
    """List up to MAX_FILES project files."""
    return _list_files(directory, root, max_files=MAX_FILES)


def list_repository_files(
    directory: str = ".",
    root: Path = PROJECT_ROOT,
) -> list[str]:
    """List all eligible project files without a file-count limit."""
    return _list_files(directory, root, max_files=None)


def read_file(
    path: str,
    root: Path = PROJECT_ROOT,
) -> str:
    """Read a bounded-size UTF-8 text file inside the project."""
    target = _resolve_safe_path(path, root)

    if not target.is_file():
        raise ValueError("File does not exist.")

    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File is too large to read.")

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File is not valid UTF-8 text.") from exc
def write_file(
    path: str,
    content: str,
    root: Path = PROJECT_ROOT,
) -> str:
    """Safely create or overwrite a UTF-8 text file."""
    target = _resolve_safe_path(path, root)

    if target.exists() and not target.is_file():
        raise ValueError("Path is not a file.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"Successfully wrote {path}."
def apply_patch(
    file_path: str,
    old_text: str,
    new_text: str,
    *,
    root: Path = PROJECT_ROOT,
) -> str:
    """Safely replace one exact text block and return the resulting diff."""
    if not file_path or not file_path.strip():
        raise ValueError("File path must not be empty.")

    if not old_text:
        raise ValueError("old_text must not be empty.")

    path = _resolve_safe_path(file_path, root)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8 text: {file_path}") from exc

    match_count = content.count(old_text)

    if match_count == 0:
        raise ValueError("Patch target was not found in the file.")

    if match_count > 1:
        raise ValueError(
            f"Patch target is ambiguous: found {match_count} matches in {file_path}."
        )

    updated_content = content.replace(old_text, new_text, 1)

    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            updated_content.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(updated_content)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return f"Successfully patched {file_path}.\n{diff}"

def search_code(
    query: str,
    directory: str = ".",
    root: Path = PROJECT_ROOT,
) -> list[dict[str, str | int]]:
    """Search text files for a case-insensitive literal string."""
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    base = _resolve_safe_path(directory, root)

    if not base.is_dir():
        raise ValueError("Directory does not exist.")

    matches = []
    needle = query.casefold()

    for path in base.rglob("*"):
        relative = path.relative_to(root.resolve())

        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        if path.stat().st_size > MAX_FILE_BYTES:
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(lines, start=1):
            if needle in line.casefold():
                matches.append(
                    {
                        "file": relative.as_posix(),
                        "line": line_number,
                        "text": line.strip()[:500],
                    }
                )

                if len(matches) >= MAX_MATCHES:
                    return matches

    return matches

def run_tests(
    test_path: str = ".",
    root: Path = PROJECT_ROOT,
) -> dict[str, str | int]:
    """Run pytest for a project path inside the allowed project root."""
    target = _resolve_safe_path(test_path, root)

    if not target.exists():
        raise ValueError("Test path does not exist.")

    command = [sys.executable, "-m", "pytest", test_path]

    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )

    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }