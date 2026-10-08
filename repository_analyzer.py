from collections import Counter
from pathlib import Path
from typing import Any

from tools import PROJECT_ROOT, list_files


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

CONFIG_FILES = {
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    ".env.example",
    "pytest.ini",
    "mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "tsconfig.json",
    "webpack.config.js",
    "vite.config.js",
    "vite.config.ts",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
}

DEPENDENCY_FILES = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
}

ENTRY_POINT_NAMES = {
    "main.py",
    "app.py",
    "manage.py",
    "main.js",
    "main.ts",
    "index.js",
    "index.ts",
    "server.js",
    "server.ts",
}


def analyze_repository(
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build a filename-based inventory of a project repository."""
    root = root.resolve()

    if not root.is_dir():
        raise ValueError("Repository root must be an existing directory.")

    files = list_files(root=root)

    language_counts: Counter[str] = Counter()
    config_files: list[str] = []
    dependency_files: list[str] = []
    test_files: list[str] = []
    source_files: list[str] = []
    entry_points: list[str] = []

    for relative_path in files:
        path = Path(relative_path)
        filename = path.name
        filename_lower = filename.lower()
        suffix = path.suffix.lower()

        if suffix in LANGUAGE_BY_EXTENSION:
            language_counts[LANGUAGE_BY_EXTENSION[suffix]] += 1
            source_files.append(relative_path)

        if filename_lower in CONFIG_FILES:
            config_files.append(relative_path)

        if filename_lower in DEPENDENCY_FILES:
            dependency_files.append(relative_path)

        path_parts = {part.lower() for part in path.parts}

        is_test_file = (
            filename_lower.startswith("test_")
            or filename_lower.endswith("_test.py")
            or ".test." in filename_lower
            or ".spec." in filename_lower
        )

        is_in_test_directory = bool({"test", "tests"} & path_parts)

        if filename_lower != "__init__.py" and (is_in_test_directory or is_test_file):
            test_files.append(relative_path)

        if filename_lower in ENTRY_POINT_NAMES:
            entry_points.append(relative_path)

    return {
        "root": str(root),
        "total_files": len(files),
        "languages": dict(sorted(language_counts.items())),
        "source_files": source_files,
        "configuration_files": config_files,
        "dependency_files": dependency_files,
        "test_files": test_files,
        "entry_points": entry_points,
        "files": files,
    }


def format_repository_summary(
    inventory: dict[str, Any],
) -> str:
    """Format a repository inventory as a readable summary."""
    languages = inventory["languages"]

    if languages:
        language_summary = ", ".join(
            f"{name} ({count} files)" for name, count in languages.items()
        )
    else:
        language_summary = "No recognized languages"

    def format_list(items: list[str]) -> str:
        if not items:
            return "- None detected"

        return "\n".join(f"- {item}" for item in items)

    return (
        "Repository Analysis\n"
        "===================\n"
        f"Root: {inventory['root']}\n"
        f"Total files: {inventory['total_files']}\n"
        f"Languages: {language_summary}\n\n"
        "Entry points:\n"
        f"{format_list(inventory['entry_points'])}\n\n"
        "Configuration files:\n"
        f"{format_list(inventory['configuration_files'])}\n\n"
        "Dependency manifests:\n"
        f"{format_list(inventory['dependency_files'])}\n\n"
        "Test files:\n"
        f"{format_list(inventory['test_files'])}"
    )
