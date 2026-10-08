from debugger import diagnose_failure, diagnose_test_result

def test_debugger_diagnoses_a_failing_test():
    """Debugger should identify the likely cause of a test failure."""
    failure = {
        "test": "test_example",
        "file": "test_example.py",
        "error": "AssertionError: assert 2 + 2 == 5",
    }

    diagnosis = diagnose_failure(failure)

    assert diagnosis["test"] == "test_example"
    assert diagnosis["cause"] == "Incorrect expected value."
import pytest

from debugger import diagnose_failure


@pytest.mark.parametrize(
    ("error", "category"),
    [
        ("AssertionError: assert 2 + 2 == 5", "assertion"),
        ("NameError: name 'foo' is not defined", "name_error"),
        ("ModuleNotFoundError: No module named 'foo'", "import_error"),
        ("SyntaxError: invalid syntax", "syntax_error"),
        ("TypeError: unsupported operand type", "type_error"),
        ("RuntimeError: something went wrong", "unknown"),
    ],
)
def test_diagnose_failure_classifies_common_errors(error, category):
    diagnosis = diagnose_failure(
        {
            "test": "test_example",
            "file": "example.py",
            "error": error,
        }
    )

    assert diagnosis["category"] == category
    assert diagnosis["error"] == error

def test_diagnose_test_result_reports_success():
    result = diagnose_test_result(
        {
            "return_code": 0,
            "stdout": "1 passed",
            "stderr": "",
        }
    )

    assert result["status"] == "passed"
    assert result["cause"] is None


def test_diagnose_test_result_reports_failure():
    result = diagnose_test_result(
        {
            "return_code": 1,
            "stdout": "AssertionError: assert 2 + 2 == 5",
            "stderr": "",
        }
    )

    assert result["category"] == "assertion"
    assert result["cause"] == "Incorrect expected value."