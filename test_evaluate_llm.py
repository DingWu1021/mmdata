import unittest

from evaluate_llm import (
    EvaluationScores,
    aggregate_scores,
    compute_scores_from_judge,
    parse_json_object,
)


class EvaluateLlmTest(unittest.TestCase):
    def test_parse_json_object_accepts_markdown_fenced_json(self):
        raw = """```json
        {"exact_success": true, "component_correct": [true, false, true]}
        ```"""

        self.assertEqual(
            {"exact_success": True, "component_correct": [True, False, True]},
            parse_json_object(raw),
        )

    def test_component_partial_score_counts_correct_intermediate_entities(self):
        scores = compute_scores_from_judge(
            {"exact_success": True, "component_correct": [True, False, True]},
            expected_components=["Boyaca", "Casanare", "Cundinamarca"],
        )

        self.assertEqual(EvaluationScores(1.0, 2 / 3), scores)

    def test_component_partial_score_falls_back_to_exact_when_no_components(self):
        correct = compute_scores_from_judge(
            {"exact_success": True, "component_correct": []},
            expected_components=[],
        )
        incorrect = compute_scores_from_judge(
            {"exact_success": False, "component_correct": []},
            expected_components=[],
        )

        self.assertEqual(EvaluationScores(1.0, 1.0), correct)
        self.assertEqual(EvaluationScores(0.0, 0.0), incorrect)

    def test_aggregate_scores_averages_exact_and_component_metrics(self):
        summary = aggregate_scores(
            [
                EvaluationScores(exact_success=1.0, component_partial=2 / 3),
                EvaluationScores(exact_success=0.0, component_partial=1 / 3),
            ]
        )

        self.assertEqual(2, summary["num_examples"])
        self.assertAlmostEqual(0.5, summary["exact_success_rate"])
        self.assertAlmostEqual(0.5, summary["component_partial_score"])


if __name__ == "__main__":
    unittest.main()
