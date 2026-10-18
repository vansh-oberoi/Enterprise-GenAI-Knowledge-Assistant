from eval.run_eval import score_case


def test_score_case_computes_hit_recall_and_reciprocal_rank():
    result = score_case(
        {"hr_policy.txt", "security_policy.txt"},
        ["unrelated.txt", "security_policy.txt", "hr_policy.txt"],
    )

    assert result["hit_at_k"] == 1
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank"] == 0.5


def test_score_case_returns_zero_when_no_expected_source_is_retrieved():
    result = score_case({"hr_policy.txt"}, ["security_policy.txt", "faq.md"])

    assert result["hit_at_k"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_unanswerable_case_is_excluded_from_retrieval_ranking_metrics():
    result = score_case(set(), ["unrelated.txt"])

    assert result == {
        "hit_at_k": None,
        "recall_at_k": None,
        "reciprocal_rank": None,
    }
