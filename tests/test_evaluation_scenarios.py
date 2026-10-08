from evaluation_scenarios import run_evaluation_scenarios


def test_all_evaluation_scenarios_match_expectations():
    results = run_evaluation_scenarios()

    assert len(results) == 3

    for result in results:
        assert result["passed"] == result["expected_pass"]


def test_evaluation_scenarios_have_unique_names():
    results = run_evaluation_scenarios()

    names = [result["name"] for result in results]

    assert len(names) == len(set(names))


def test_evaluation_scenarios_include_failure_case():
    results = run_evaluation_scenarios()

    failure = next(
        result for result in results if result["name"] == "unrecoverable failure"
    )

    assert failure["passed"] is False
    assert failure["expected_pass"] is False
