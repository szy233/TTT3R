"""Tests for the update-rule variants used by the ICLR 2027 rebuttal experiments.

Run from the repository root:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m unittest rebuttal/iclr2027/tests/test_update_rules.py
"""
import copy
import types
import unittest

import numpy as np
import torch

from dust3r.model import ARCroco3DStereo as Model

ALPHA_PERP = 0.5
ALPHA_PARALLEL = 0.05
BETA = 0.95


def make_config(**overrides):
    values = dict(
        alpha_perp=ALPHA_PERP,
        alpha_parallel=ALPHA_PARALLEL,
        beta_ema=BETA,
        gamma=0.0,
        auto_gamma="",
        ortho_adaptive="",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def prepared_step(config, history=3, seed=0):
    """Advance the tracker for a few frames and return the inputs of the next frame."""
    torch.manual_seed(seed)
    state = torch.randn(1, 6, 8, dtype=torch.float64)
    tracker = {}
    for _ in range(history):
        candidate = state + torch.randn_like(state)
        state = Model._delta_ortho_update(state, candidate, tracker, config)
    candidate = state + torch.randn_like(state)
    return state, candidate, tracker


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def split(delta, direction):
    parallel = (delta * direction).sum(dim=-1, keepdim=True) * direction
    return parallel, delta - parallel


class DefaultRule(unittest.TestCase):
    def test_default_projects_onto_the_updated_ema_direction(self):
        config = make_config()
        state, candidate, tracker = prepared_step(config)
        previous = tracker["drift_dir"].clone()
        delta = candidate - state
        updated = unit(BETA * previous + (1 - BETA) * unit(delta))
        parallel, perpendicular = split(delta, updated)

        result = Model._delta_ortho_update(state, candidate, tracker, config)

        expected = state + ALPHA_PERP * perpendicular + ALPHA_PARALLEL * parallel
        self.assertTrue(torch.allclose(result, expected))

    def test_beta_zero_makes_the_reference_the_current_delta(self):
        # Equation (7) includes the current delta, so at beta_ema = 0 the whole
        # update is parallel. This is the behaviour MDAW points out against A.8.
        config = make_config(beta_ema=0.0)
        state, candidate, tracker = prepared_step(config)

        result = Model._delta_ortho_update(state, candidate, tracker, config)

        expected = state + ALPHA_PARALLEL * (candidate - state)
        self.assertTrue(torch.allclose(result, expected))


class NormMatch(unittest.TestCase):
    def outputs(self, **overrides):
        base = make_config()
        state, candidate, tracker = prepared_step(base)
        ortho = Model._delta_ortho_update(state, candidate, copy.deepcopy(tracker), base)
        matched = Model._delta_ortho_update(
            state, candidate, copy.deepcopy(tracker), make_config(ortho_norm_match=True, **overrides)
        )
        return state, candidate, ortho, matched

    def test_per_token_norm_equals_the_directional_rule(self):
        state, _, ortho, matched = self.outputs()
        self.assertTrue(
            torch.allclose((matched - state).norm(dim=-1), (ortho - state).norm(dim=-1))
        )

    def test_update_keeps_the_direction_of_the_raw_delta(self):
        state, candidate, _, matched = self.outputs()
        cosine = torch.nn.functional.cosine_similarity(matched - state, candidate - state, dim=-1)
        self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine)))

    def test_equal_alphas_reduce_to_constant_dampening(self):
        config = make_config(alpha_perp=0.5, alpha_parallel=0.5, ortho_norm_match=True)
        state, candidate, tracker = prepared_step(config)
        result = Model._delta_ortho_update(state, candidate, tracker, config)
        self.assertTrue(torch.allclose(result, state + 0.5 * (candidate - state)))


class ProjectPrevious(unittest.TestCase):
    def test_decomposes_against_the_direction_before_this_frame(self):
        config = make_config(ortho_project_prev=True)
        state, candidate, tracker = prepared_step(config)
        previous = tracker["drift_dir"].clone()
        parallel, perpendicular = split(candidate - state, previous)

        result = Model._delta_ortho_update(state, candidate, tracker, config)

        expected = state + ALPHA_PERP * perpendicular + ALPHA_PARALLEL * parallel
        self.assertTrue(torch.allclose(result, expected))

    def test_the_tracker_still_advances_with_the_current_delta(self):
        config = make_config(ortho_project_prev=True)
        state, candidate, tracker = prepared_step(config)
        previous = tracker["drift_dir"].clone()
        delta = candidate - state

        Model._delta_ortho_update(state, candidate, tracker, config)

        expected = unit(BETA * previous + (1 - BETA) * unit(delta))
        self.assertTrue(torch.allclose(tracker["drift_dir"], expected))


class GainLog(unittest.TestCase):
    def test_records_the_mean_per_token_gain(self):
        log = []
        config = make_config(gain_log=log)
        state, candidate, tracker = prepared_step(make_config())

        result = Model._delta_ortho_update(state, candidate, tracker, config)

        gain = (result - state).norm(dim=-1) / (candidate - state).norm(dim=-1)
        self.assertEqual(len(log), 1)
        self.assertAlmostEqual(log[0], gain.mean().item(), places=9)


class GateProbe(unittest.TestCase):
    def setUp(self):
        self.gate = torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1)

    def test_without_a_probe_the_gate_is_returned_unchanged(self):
        self.assertIs(Model._gate_probe(self.gate, None), self.gate)

    def test_records_each_token_gate_as_a_vector(self):
        probe = {"log": []}
        result = Model._gate_probe(self.gate, probe)
        self.assertIs(result, self.gate)
        self.assertTrue(torch.equal(probe["log"][0], torch.tensor([0.2, 0.4, 0.6])))

    def test_frozen_gate_replaces_the_live_one(self):
        frozen = torch.tensor([0.1, 0.1, 0.9]).view(1, 3, 1)
        result = Model._gate_probe(self.gate, {"frozen": frozen})
        self.assertTrue(torch.equal(result, frozen))

    def test_frozen_gate_is_the_time_mean_of_each_token(self):
        log = np.array([[0.2, 0.4], [0.4, 0.8]], dtype=np.float32)
        frozen = Model._frozen_gate_from_log(log)
        self.assertEqual(frozen.shape, torch.Size([1, 2, 1]))
        self.assertTrue(torch.allclose(frozen.view(-1), torch.tensor([0.3, 0.6])))


if __name__ == "__main__":
    unittest.main()
