"""The comparison rule registered in docs/PREREG_E1_E5.md, shared by every ruling script."""
import itertools
import math
import os
import random
import re

import numpy as np

NOISE_FLOOR = 0.022  # Largest gap between two nominally identical configurations.
FROZEN_MARGIN = 0.05  # E3: how much freezing may cost, as a share of the live mean.

_SEQ = re.compile(r"^(\S+?)\s*\|\s*ATE:\s*([0-9.]+)")
_ATE_FIELD = r"APE w\.r\.t\. translation part.*?\b{}\s+([0-9.]+)"
_METRIC_FILE = re.compile(r"^(.+)_eval_metric\.txt$")


def load_ate(folder):
    """Per-sequence ATE RMSE from a run's _error_log.txt, keyed by sequence name."""
    out = {}
    with open(os.path.join(folder, "_error_log.txt"), errors="ignore") as handle:
        for line in handle:
            match = _SEQ.match(line)
            if match:
                out[match.group(1).split("-", 1)[-1]] = float(match.group(2))
    return out


def _load_metric_field(folder, field):
    pattern = re.compile(_ATE_FIELD.format(field), re.DOTALL)
    out = {}
    for name in sorted(os.listdir(folder)):
        file_match = _METRIC_FILE.match(name)
        if not file_match:
            continue
        with open(os.path.join(folder, name), errors="ignore") as handle:
            block = pattern.search(handle.read())
        if block:
            out[file_match.group(1)] = float(block.group(1))
    return out


def load_ate_mean(folder):
    """Per-sequence ATE mean from a run's <seq>_eval_metric.txt files."""
    return _load_metric_field(folder, "mean")


def load_ate_rmse(folder):
    """Per-sequence ATE RMSE from <seq>_eval_metric.txt, for runs with no usable _error_log.txt."""
    return _load_metric_field(folder, "rmse")


def wilcoxon(diffs):
    """Two-sided signed-rank p, exact for n <= 16 and normal above that."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ordered = sorted(abs(x) for x in d)
    rank = {}
    i = 0
    while i < n:
        j = i
        while j < n and ordered[j] == ordered[i]:
            j += 1
        rank[ordered[i]] = (i + j + 1) / 2
        i = j
    r = [rank[abs(x)] for x in d]
    w = sum(ri for ri, x in zip(r, d) if x > 0)
    if n <= 16:
        total = sum(r)
        low = min(w, total - w)
        hits = sum(
            sum(ri for ri, b in zip(r, signs) if b) <= low + 1e-9
            for signs in itertools.product((0, 1), repeat=n)
        )
        return min(1.0, 2 * hits / 2 ** n)
    mu = n * (n + 1) / 4
    sd = math.sqrt(sum(x * x for x in r) / 4)
    return math.erfc(abs((w - mu) / sd) / math.sqrt(2))


def bootstrap(diffs, resamples=20000, seed=0):
    """Paired bootstrap 95% CI of the mean difference."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def common_rule(mean_a, mean_b, p):
    """Which side wins: "a", "b" or "tied". The gap is relative to the smaller mean."""
    if p >= 0.05 or abs(mean_a - mean_b) / min(mean_a, mean_b) <= NOISE_FLOOR:
        return "tied"
    return "a" if mean_a < mean_b else "b"


def frozen_verdict(live, frozen, ci_hi, p):
    """E3 verdict. ci_hi is the upper bound of (frozen - live).

    The registration does not say what happens if both criteria hold. Refuted takes
    precedence here, since it is the only reading that cannot overstate the prediction.
    """
    if common_rule(frozen, live, p) == "b":
        return "refuted"
    return "supported" if ci_hi <= FROZEN_MARGIN * live else "inconclusive"


def gate_variance_parts(gate):
    """E3a. Split a gate [frames, tokens] into token, frame and residual variance.

    The residual is the token-level temporal part: what a token does over time beyond
    its own mean and the frame's mean. The two stds compare that view with the
    frame-averaged one the paper reports.
    """
    gate = np.asarray(gate, dtype=np.float64)
    token_mean = gate.mean(axis=0)
    frame_mean = gate.mean(axis=1)
    residual = gate - token_mean[None, :] - frame_mean[:, None] + gate.mean()
    return dict(
        total=float(gate.var()),
        token=float(token_mean.var()),
        frame=float(frame_mean.var()),
        residual=float(residual.var()),
        token_temporal_std=float(gate.std(axis=0).mean()),
        frame_mean_std=float(frame_mean.std()),
    )


def holm(pvalues):
    """Holm step-down adjusted p-values, returned in the order given."""
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def gap_shrink(ortho_a, replay_a, ortho_b, replay_b):
    """E6c. Mean and paired bootstrap CI of (ortho_a - replay_a) - (ortho_b - replay_b)."""
    keys = sorted(set(ortho_a) & set(replay_a) & set(ortho_b) & set(replay_b))
    d = [(ortho_a[k] - replay_a[k]) - (ortho_b[k] - replay_b[k]) for k in keys]
    return sum(d) / len(d), bootstrap(d)


def compare(a, b):
    """Paired comparison of two {sequence: value} maps over their shared sequences."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    mean_a = sum(a[k] for k in keys) / len(keys)
    mean_b = sum(b[k] for k in keys) / len(keys)
    lo, hi = bootstrap(d)
    p = wilcoxon(d)
    return dict(
        n=len(keys),
        mean_a=mean_a,
        mean_b=mean_b,
        gap=mean_a / mean_b - 1,
        wins_a=sum(x < 0 for x in d),
        ci=(lo, hi),
        p=p,
        verdict=common_rule(mean_a, mean_b, p),
    )
