import argparse
from pathlib import Path

from code_indexer import build_code_index
from code_retriever import retrieve_code
from embeddings import DEFAULT_EMBEDDING_MODEL
from index_manifest import (
    build_source_manifest,
    compare_manifests,
    format_manifest_changes,
)
from tools import PROJECT_ROOT
from vector_store import VectorStore


def _is_valid_source_manifest(manifest):
    """Check that a source manifest has the expected structure."""
    if not isinstance(manifest, dict):
        return False

    for file_path, details in manifest.items():
        if not isinstance(file_path, str) or not file_path.strip():
            return False

        if not isinstance(details, dict):
            return False

        for field in ("size", "mtime_ns"):
            value = details.get(field)
            if type(value) is not int or value < 0:
                return False

    return True


def _is_valid_exclusion_list(value):
    """Check that saved exclusion metadata is a list of nonempty strings."""
    return isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def check_index_freshness(store, root):
    """Report whether a saved code index may be stale."""
    metadata = store.metadata

    saved_manifest = metadata.get("source_manifest")

    if saved_manifest is None:
        print(
            "Warning: This index has no source manifest. Use --rebuild to refresh it."
        )
        return

    if not _is_valid_source_manifest(saved_manifest):
        print(
            "Warning: The index has an invalid source manifest. "
            "Use --rebuild to refresh it."
        )
        return

    saved_root = metadata.get("root")
    if not isinstance(saved_root, str) or saved_root != str(root.resolve()):
        print(
            "Warning: The index was built for a different repository root "
            "or has invalid root metadata. Use --rebuild to refresh it."
        )
        return

    raw_exclude_dirs = metadata.get("exclude_dirs", [])
    raw_exclude_files = metadata.get("exclude_files", [])

    if not _is_valid_exclusion_list(raw_exclude_dirs):
        print(
            "Warning: The index has invalid directory exclusion metadata. "
            "Use --rebuild to refresh it."
        )
        return

    if not _is_valid_exclusion_list(raw_exclude_files):
        print(
            "Warning: The index has invalid file exclusion metadata. "
            "Use --rebuild to refresh it."
        )
        return

    current_manifest = build_source_manifest(
        root,
        exclude_dirs=set(raw_exclude_dirs),
        exclude_files=set(raw_exclude_files),
    )

    changes = compare_manifests(saved_manifest, current_manifest)
    messages = format_manifest_changes(changes)

    if messages:
        print("Warning: The code index may be stale.")

        for message in messages:
            print(message)

        print("Run with --rebuild to refresh the index.")
    else:
        print("Index freshness check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search a codebase using semantic code retrieval."
    )

    parser.add_argument(
        "question",
        help="A natural-language question about the codebase.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository directory to index.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Maximum number of chunks to embed when building the index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matching chunks to display.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Path to the saved index file.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild and overwrite the saved index.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory name to exclude when building the index. Repeatable.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Repository-relative file path to exclude when building the index. "
            "Repeatable."
        ),
    )

    args = parser.parse_args()

    if args.max_chunks <= 0:
        parser.error("--max-chunks must be greater than zero.")

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    if not args.root.is_dir():
        parser.error(f"Repository directory does not exist: {args.root}")

    root = args.root.resolve()
    index_path = (
        args.index_path.resolve()
        if args.index_path is not None
        else root / ".code_index.json"
    )

    model = DEFAULT_EMBEDDING_MODEL

    if index_path.exists() and not args.rebuild:
        if args.exclude_dir or args.exclude_file:
            print(
                "Note: exclusion options apply only when building the index. "
                "Use --rebuild to apply them."
            )

        print(f"Loading saved index: {index_path}")

        try:
            store = VectorStore.load(
                index_path,
                expected_model=model,
            )
            check_index_freshness(store, root)
        except (ValueError, OSError) as exc:
            parser.error(
                f"Cannot load saved index: {exc} "
                "Check the index file or use --rebuild to create a new index."
            )
    else:
        print(f"Indexing repository: {root}")
        print(f"Maximum chunks to embed: {args.max_chunks}")

        if args.exclude_dir:
            print(f"Excluded directories: {', '.join(args.exclude_dir)}")

        if args.exclude_file:
            print(f"Excluded files: {', '.join(args.exclude_file)}")

        store = build_code_index(
            root,
            max_chunks=args.max_chunks,
            model=model,
            exclude_dirs=set(args.exclude_dir),
            exclude_files=set(args.exclude_file),
        )

        index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(index_path, model=model)

        print(f"Saved index: {index_path}")

    if len(store) == 0:
        print("No supported code chunks were found.")
        return 0

    print(f"Indexed chunks: {len(store)}")
    print(f"\nQuestion: {args.question}\n")

    results = retrieve_code(
        args.question,
        store,
        top_k=args.top_k,
        model=model,
    )

    if not results:
        print("No matching code chunks found.")
        return 0

    print("Search results")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result['file_path']}:"
            f"{result['start_line']}-{result['end_line']}"
        )
        print(f"   Language: {result['language']}")
        print(f"   Similarity: {result['score']:.4f}")
        print("-" * 60)
        print(result["content"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
