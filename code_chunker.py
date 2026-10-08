from pathlib import Path
from typing import Any

from tools import (
    IGNORED_DIRS,
    MAX_FILE_BYTES,
    PROJECT_ROOT,
    list_repository_files,
    read_file,
)


CHUNK_SIZE = 80
CHUNK_OVERLAP = 15
MAX_CHUNKS = 2_000

LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".h": "C/C++",
    ".cs": "C#",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".sh": "Shell",
}


def chunk_text(
    content: str,
    *,
    file_path: str,
    language: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split text into overlapping line-based chunks with source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    lines = content.splitlines()

    if not lines:
        return []

    chunks = []
    step = chunk_size - overlap

    for start_index in range(0, len(lines), step):
        end_index = min(start_index + chunk_size, len(lines))

        chunks.append(
            {
                "file_path": file_path,
                "start_line": start_index + 1,
                "end_line": end_index,
                "language": language,
                "content": "\n".join(lines[start_index:end_index]),
            }
        )

        if end_index == len(lines):
            break

    return chunks

def chunk_files(
    file_paths: set[str],
    root: Path = PROJECT_ROOT,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Chunk only the specified eligible source files."""
    root = root.resolve()

    if not root.is_dir():
        raise ValueError("Repository root must be an existing directory.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    excluded_dirs = IGNORED_DIRS | (exclude_dirs or set())
    excluded_files = {
        Path(file_path).as_posix()
        for file_path in (exclude_files or set())
    }

    chunks: list[dict[str, Any]] = []

    for relative_path in sorted(file_paths):
        path = Path(relative_path)
        normalized_path = path.as_posix()

        if path.is_absolute() or ".." in path.parts:
            continue

        if normalized_path in excluded_files:
            continue

        if any(part in excluded_dirs for part in path.parts[:-1]):
            continue

        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if language is None:
            continue

        try:
            content = read_file(normalized_path, root=root)
        except (ValueError, OSError):
            continue

        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            continue

        chunks.extend(
            chunk_text(
                content,
                file_path=normalized_path,
                language=language,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks


def chunk_repository_with_status(
    root: Path = PROJECT_ROOT,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Return repository chunks and whether the limit truncated indexing."""
    root = root.resolve()

    if not root.is_dir():
        raise ValueError("Repository root must be an existing directory.")

    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    excluded_dirs = IGNORED_DIRS | (exclude_dirs or set())
    excluded_files = {
        Path(file_path).as_posix() for file_path in (exclude_files or set())
    }

    chunks: list[dict[str, Any]] = []

    for relative_path in list_repository_files(root=root):
        path = Path(relative_path)
        normalized_path = path.as_posix()

        if normalized_path in excluded_files:
            continue

        if any(part in excluded_dirs for part in path.parts[:-1]):
            continue

        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if language is None:
            continue

        try:
            content = read_file(relative_path, root=root)
        except (ValueError, OSError):
            continue

        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            continue

        file_chunks = chunk_text(
            content,
            file_path=relative_path,
            language=language,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        remaining = max_chunks - len(chunks)

        if len(file_chunks) > remaining:
            chunks.extend(file_chunks[:remaining])
            return chunks, True

        chunks.extend(file_chunks)

        if len(chunks) == max_chunks:
            # Continue scanning eligible files to determine whether
            # any additional chunks actually exist.
            continue

    return chunks, False


def chunk_repository(
    root: Path = PROJECT_ROOT,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS,
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Chunk a repository while preserving the original list-returning API."""
    chunks, _ = chunk_repository_with_status(
        root,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks=max_chunks,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
    )
    return chunks
