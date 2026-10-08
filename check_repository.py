from repository_analyzer import (
    analyze_repository,
    format_repository_summary,
)

inventory = analyze_repository()
print(format_repository_summary(inventory))
