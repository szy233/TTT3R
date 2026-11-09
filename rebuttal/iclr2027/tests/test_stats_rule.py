"""Tests for the registered comparison rule in docs/PREREG_E1_E5.md.

Run from the repository root:
    python3 -m unittest rebuttal/iclr2027/tests/test_stats_rule.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from stats_rule import (
    bootstrap,
    compare,
    common_rule,
    frozen_verdict,
    gap_shrink,
    gate_variance_parts,
    holm,
    load_ate,
    load_ate_mean,
    load_ate_rmse,
    wilcoxon,
)


class Holm(unittest.TestCase):
    def test_smallest_p_is_multiplied_by_the_number_of_tests(self):
        self.assertAlmostEqual(holm([0.01, 0.04, 0.03])[0], 0.03)

    def test_adjusted_values_never_decrease_with_rank(self):
        self.assertEqual(holm([0.001, 0.02, 0.03]), [0.003, 0.04, 0.04])

    def test_adjusted_values_stop_at_one(self):
        self.assertEqual(holm([0.6, 0.7]), [1.0, 1.0])

    def test_results_keep_the_input_order(self):
        adjusted = holm([0.04, 0.01])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.02)


class GapShrink(unittest.TestCase):
    """E6c: did the harm of direction fall between two settings of the parallel gain?"""

    def test_harm_that_falls_on_every_sequence_gives_an_interval_above_zero(self):
        keys = [str(i) for i in range(20)]
        ortho_a = {k: 1.0 + 0.01 * i for i, k in enumerate(keys)}
        replay_a = {k: 0.5 for k in keys}
        ortho_b = {k: 0.7 + 0.01 * i for i, k in enumerate(keys)}
        replay_b = {k: 0.5 for k in keys}
        mean, (lo, hi) = gap_shrink(ortho_a, replay_a, ortho_b, replay_b)
        self.assertAlmostEqual(mean, 0.3)
        self.assertGreater(lo, 0)
        self.assertGreaterEqual(hi, lo)

    def test_unchanged_harm_is_not_reported_as_shrinking(self):
        keys = [str(i) for i in range(20)]
        ortho = {k: 1.0 + 0.01 * i for i, k in enumerate(keys)}
        replay = {k: 0.5 for k in keys}
        mean, (lo, _) = gap_shrink(ortho, replay, ortho, replay)
        self.assertEqual(mean, 0)
        self.assertLessEqual(lo, 0)

    def test_only_sequences_present_in_all_four_runs_count(self):
        a = {"x": 2.0, "y": 9.0}
        b = {"x": 1.0, "y": 1.0}
        c = {"x": 1.5}
        d = {"x": 1.0}
        mean, _ = gap_shrink(a, b, c, d)
        self.assertAlmostEqual(mean, 0.5)


class CommonRule(unittest.TestCase):
    def test_a_wins_with_lower_mean_small_p_and_a_gap_above_the_floor(self):
        self.assertEqual(common_rule(1.00, 1.10, 0.01), "a")

    def test_b_wins_when_the_lower_mean_is_b(self):
        self.assertEqual(common_rule(1.10, 1.00, 0.01), "b")

    def test_tied_when_p_is_not_below_005(self):
        self.assertEqual(common_rule(1.00, 1.10, 0.05), "tied")

    def test_tied_when_the_gap_is_inside_the_noise_floor(self):
        self.assertEqual(common_rule(1.000, 1.021, 0.01), "tied")

    def test_gap_is_relative_to_the_smaller_mean_as_in_paired_stats(self):
        # 1.0 against 1.0222 is 2.22% of the smaller mean and 2.17% of the larger one.
        self.assertEqual(common_rule(1.0, 1.0222, 0.01), "a")


class Wilcoxon(unittest.TestCase):
    def test_all_eight_on_one_side_gives_the_exact_two_sided_minimum(self):
        self.assertAlmostEqual(wilcoxon([-1, -2, -3, -4, -5, -6, -7, -8]), 2 / 256)

    def test_eleven_on_one_side_gives_the_exact_two_sided_minimum(self):
        self.assertAlmostEqual(wilcoxon([-float(i) for i in range(1, 12)]), 2 / 2048)

    def test_balanced_differences_give_p_one(self):
        self.assertEqual(wilcoxon([-1, 1, -2, 2]), 1.0)


class Bootstrap(unittest.TestCase):
    def test_interval_brackets_the_mean_and_repeats_with_the_same_seed(self):
        d = [-3.0, -1.0, -2.0, 0.5, -1.5, -2.5, -0.5, -1.0]
        lo, hi = bootstrap(d)
        self.assertLess(lo, sum(d) / len(d))
        self.assertGreater(hi, sum(d) / len(d))
        self.assertEqual((lo, hi), bootstrap(d))


class FrozenVerdict(unittest.TestCase):
    def test_supported_when_upper_bound_is_within_five_percent_of_live(self):
        self.assertEqual(frozen_verdict(live=1.0, frozen=1.0, ci_hi=0.05, p=0.5), "supported")

    def test_inconclusive_when_upper_bound_exceeds_five_percent_and_not_worse(self):
        self.assertEqual(frozen_verdict(live=1.0, frozen=1.0, ci_hi=0.06, p=0.5), "inconclusive")

    def test_refuted_when_frozen_loses_by_the_common_rule(self):
        self.assertEqual(frozen_verdict(live=1.0, frozen=1.2, ci_hi=0.3, p=0.01), "refuted")

    def test_refuted_wins_when_both_criteria_hold(self):
        # A tight interval can sit under 5% while frozen still loses by the common rule.
        self.assertEqual(frozen_verdict(live=1.0, frozen=1.03, ci_hi=0.04, p=0.01), "refuted")


class GateVariance(unittest.TestCase):
    def test_a_gate_that_differs_only_between_tokens_has_no_temporal_part(self):
        gate = [[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]]
        parts = gate_variance_parts(gate)
        self.assertAlmostEqual(parts["token"], 0.01)
        self.assertAlmostEqual(parts["frame"], 0.0)
        self.assertAlmostEqual(parts["residual"], 0.0)
        self.assertAlmostEqual(parts["token_temporal_std"], 0.0)
        self.assertAlmostEqual(parts["frame_mean_std"], 0.0)

    def test_a_gate_that_moves_all_tokens_together_is_only_a_frame_effect(self):
        gate = [[0.1, 0.1], [0.3, 0.3]]
        parts = gate_variance_parts(gate)
        self.assertAlmostEqual(parts["frame"], 0.01)
        self.assertAlmostEqual(parts["token"], 0.0)
        self.assertAlmostEqual(parts["residual"], 0.0)
        self.assertAlmostEqual(parts["frame_mean_std"], 0.1)
        self.assertAlmostEqual(parts["token_temporal_std"], 0.1)

    def test_tokens_moving_in_opposite_directions_show_up_only_in_the_residual(self):
        # The frame average is flat here, which is what the paper's frame-level statistic would see.
        gate = [[0.1, 0.3], [0.3, 0.1]]
        parts = gate_variance_parts(gate)
        self.assertAlmostEqual(parts["residual"], 0.01)
        self.assertAlmostEqual(parts["frame"], 0.0)
        self.assertAlmostEqual(parts["token"], 0.0)
        self.assertAlmostEqual(parts["frame_mean_std"], 0.0)
        self.assertAlmostEqual(parts["token_temporal_std"], 0.1)

    def test_the_three_parts_add_up_to_the_total_variance(self):
        gate = [[0.2, 0.5, 0.1], [0.4, 0.1, 0.3], [0.9, 0.2, 0.6], [0.3, 0.3, 0.7]]
        parts = gate_variance_parts(gate)
        self.assertAlmostEqual(parts["token"] + parts["frame"] + parts["residual"], parts["total"])


class Logs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, text):
        with open(os.path.join(self.tmp.name, name), "w") as handle:
            handle.write(text)

    def test_load_ate_reads_each_sequence_once_and_strips_the_dataset_prefix(self):
        self._write(
            "_error_log.txt",
            "kitti_odo_full-03 | ATE: 39.90691, RPE trans: 3.6, RPE rot: 1.0\n39.90691\n3.6\n1.0\n"
            "kitti_odo_full-04 | ATE: 17.78537, RPE trans: 3.2, RPE rot: 0.5\n"
            "Average ATE: 28.84614, Average RPE trans: 3.4, Average RPE rot: 0.8\n",
        )
        self.assertEqual(load_ate(self.tmp.name), {"03": 39.90691, "04": 17.78537})

    def test_load_ate_mean_takes_the_translation_block_not_the_rotation_block(self):
        self._write(
            "04_eval_metric.txt",
            "Seq: 04\n\nAPE w.r.t. translation part (m)\n\n       max\t48.7\n       mean\t15.454203\n"
            "       rmse\t17.785367\nRPE w.r.t. rotation angle in degrees (deg)\n       mean\t0.44\n",
        )
        self.assertEqual(load_ate_mean(self.tmp.name), {"04": 15.454203})

    def test_load_ate_rmse_reads_runs_that_have_only_metric_files(self):
        self._write(
            "04_eval_metric.txt",
            "Seq: 04\n\nAPE w.r.t. translation part (m)\n\n       mean\t6.137777\n       rmse\t7.759798\n"
            "RPE w.r.t. rotation angle in degrees (deg)\n       rmse\t0.51\n",
        )
        self.assertEqual(load_ate_rmse(self.tmp.name), {"04": 7.759798})


class Compare(unittest.TestCase):
    def test_compare_uses_only_shared_sequences(self):
        a = {"x": 1.0, "y": 2.0, "z": 9.0}
        b = {"x": 2.0, "y": 3.0}
        result = compare(a, b)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean_a"], 1.5)
        self.assertEqual(result["mean_b"], 2.5)
        self.assertEqual(result["wins_a"], 2)
        self.assertEqual(result["verdict"], "tied")  # n=2 cannot reach p < 0.05


if __name__ == "__main__":
    unittest.main()
