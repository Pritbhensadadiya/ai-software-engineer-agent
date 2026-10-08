def review_code(change: dict) -> dict:
    """Review a code change and return structured findings."""
    code = change.get("code", "")

    findings = []

    if "raise Exception(" in code:
        findings.append(
            {
                "severity": "error",
                "message": (
                    "Generic Exception should be avoided; use a specific exception."
                ),
            }
        )

    if "return None" in code:
        findings.append(
            {
                "severity": "warning",
                "message": ("Function returns None; verify that this is intended."),
            }
        )

    if "* 0" in code:
        findings.append(
            {
                "severity": "warning",
                "message": (
                    "Multiplication by zero may discard the input value; "
                    "verify that this behavior is intentional."
                ),
            }
        )

    status = "changes_requested" if findings else "approved"

    return {
        "file": change.get("file", ""),
        "status": status,
        "findings": findings,
    }
