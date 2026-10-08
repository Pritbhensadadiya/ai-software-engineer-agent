def diagnose_failure(failure: dict) -> dict:
    """Classify a test failure and return a structured diagnosis."""
    error = failure.get("error", "")

    if "AssertionError" in error:
        cause = "Incorrect expected value."
        category = "assertion"

    elif "NameError" in error:
        cause = "A name is not defined."
        category = "name_error"

    elif "ImportError" in error or "ModuleNotFoundError" in error:
        cause = "A required module or import is unavailable."
        category = "import_error"

    elif "SyntaxError" in error:
        cause = "The code contains a syntax error."
        category = "syntax_error"

    elif "TypeError" in error:
        cause = "The code received an unexpected type or arguments."
        category = "type_error"

    else:
        cause = "The failure cause could not be classified automatically."
        category = "unknown"

    return {
        "test": failure.get("test", ""),
        "file": failure.get("file", ""),
        "category": category,
        "cause": cause,
        "error": error,
    }
def diagnose_test_result(test_result: dict) -> dict:
    """Convert a test execution result into a structured diagnosis."""
    if test_result.get("return_code") == 0:
        return {
            "status": "passed",
            "cause": None,
            "category": None,
        }

    error_text = "\n".join(
        value
        for value in (
            test_result.get("stdout", ""),
            test_result.get("stderr", ""),
        )
        if value
    )

    return diagnose_failure(
        {
            "test": "",
            "file": "",
            "error": error_text,
        }
    )