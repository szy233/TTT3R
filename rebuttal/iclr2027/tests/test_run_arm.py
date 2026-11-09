"""Each E6 arm must map to the parameters registered in docs/PREREG_E6.md.

Folder names have misled us about parameters before, so the mapping is tested directly.

Run from the repository root:
    python3 -m unittest rebuttal/iclr2027/tests/test_run_arm.py
"""
import os
import subprocess
import unittest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_arm.sh")


def dry_run(dataset, arm, tag="-"):
    """The update type and flags run_arm.sh would use, without starting a model."""
    # DDD3R_PYTHON=/bin/false means a broken dry run fails at once instead of starting a real
    # job on a GPU, which is what happened the first time this test ran.
    env = dict(os.environ, DDD3R_DRY_RUN="1", DDD3R_PYTHON="/bin/false")
    done = subprocess.run(
        ["bash", SCRIPT, "0", dataset, arm, tag],
        capture_output=True, text=True, env=env, timeout=30,
        cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
    )
    return done.returncode, done.stdout.strip()


class FlatMultipliers(unittest.TestCase):
    def test_each_arm_uses_its_registered_alpha(self):
        for arm, alpha in (
            ("const010", "0.10"), ("const015", "0.15"), ("const020", "0.20"),
            ("const025", "0.25"), ("const033", "0.33"), ("const05", "0.5"),
        ):
            with self.subTest(arm=arm):
                code, out = dry_run("tum_s1_1000", arm)
                self.assertEqual(code, 0)
                self.assertIn("TYPE=ddd3r_constant", out)
                self.assertIn(f"--alpha {alpha}", out.split("EXTRA=")[1] + " ")


class RelaxedParallelGain(unittest.TestCase):
    def test_logging_arm_uses_alpha_parallel_015_and_its_own_norm_folder(self):
        code, out = dry_run("tum_s1_1000", "ortho_a015")
        self.assertEqual(code, 0)
        self.assertIn("TYPE=ddd3r ", out)
        self.assertIn("--alpha_perp 0.5 --alpha_parallel 0.15 --beta_ema 0.95 --gamma 0", out)
        self.assertIn("--norm_log_dir rebuttal/iclr2027/results/norms/tum_s1_1000__a015", out)

    def test_replay_reads_the_log_of_the_same_setting(self):
        code, out = dry_run("tum_s1_1000", "normreplay_a015")
        self.assertEqual(code, 0)
        self.assertIn("--alpha_parallel 0.15", out)
        self.assertIn("--norm_replay_dir rebuttal/iclr2027/results/norms/tum_s1_1000__a015", out)

    def test_default_setting_keeps_its_own_norm_folder(self):
        code, out = dry_run("kitti_odo_full", "ortho_check")
        self.assertEqual(code, 0)
        self.assertIn("--alpha_parallel 0.05", out)
        self.assertIn("--norm_log_dir rebuttal/iclr2027/results/norms/kitti_odo_full ", out + " ")


class Guards(unittest.TestCase):
    def test_unknown_arm_is_refused(self):
        code, _ = dry_run("tum_s1_1000", "const099")
        self.assertEqual(code, 2)


class DirectionAtFixedScale(unittest.TestCase):
    def test_each_arm_uses_the_default_ratio_at_its_registered_scale(self):
        for arm, scale in (("dir010", "0.10"), ("dir020", "0.20"), ("dir033", "0.33")):
            with self.subTest(arm=arm):
                code, out = dry_run("kitti_odo_full", arm)
                self.assertEqual(code, 0)
                self.assertIn("TYPE=ddd3r ", out)
                self.assertIn("--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0", out)
                self.assertIn(f"--ortho_scale {scale} ", out + " ")
                self.assertIn(f"--gain_log_dir rebuttal/iclr2027/results/gains/kitti_odo_full/{arm}", out)

    def test_isotropic_check_arm_has_equal_alphas(self):
        code, out = dry_run("kitti_odo_full", "iso033")
        self.assertEqual(code, 0)
        self.assertIn("--alpha_perp 0.5 --alpha_parallel 0.5", out)
        self.assertIn("--ortho_scale 0.33", out)


class PerturbedArms(unittest.TestCase):
    def test_flat_arm_moves_alpha_by_one_part_in_ten_million(self):
        code, out = dry_run("tum_s1_1000", "const033p")
        self.assertEqual(code, 0)
        self.assertIn("TYPE=ddd3r_constant", out)
        self.assertIn("--alpha 0.3300001", out)

    def test_ortho_arm_moves_only_alpha_perp(self):
        code, out = dry_run("kitti_odo_full", "orthop")
        self.assertEqual(code, 0)
        self.assertIn("TYPE=ddd3r ", out)
        self.assertIn("--alpha_perp 0.5000001 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0", out)

    def test_queue_runs_the_two_arms_on_the_two_datasets(self):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_e7_noise.sh")
        env = dict(os.environ, DDD3R_DRY_RUN="1", DDD3R_QUEUE_LOG="/dev/null", DDD3R_ECHO_ARMS="1")
        done = subprocess.run(
            ["bash", script], capture_output=True, text=True, env=env, timeout=60,
            cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        arms = sorted(line[len("ARM "):] for line in done.stdout.splitlines() if line.startswith("ARM "))
        self.assertEqual(arms, sorted(
            f"{d} {a}" for d in ("tum_s1_1000", "kitti_odo_full") for a in ("const033p", "orthop")
        ))


class E7Queue(unittest.TestCase):
    def test_queue_runs_exactly_the_registered_arms(self):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_e7.sh")
        env = dict(os.environ, DDD3R_DRY_RUN="1", DDD3R_QUEUE_LOG="/dev/null", DDD3R_ECHO_ARMS="1")
        done = subprocess.run(
            ["bash", script],
            capture_output=True, text=True, env=env, timeout=120,
            cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        arms = sorted(line[len("ARM "):] for line in done.stdout.splitlines() if line.startswith("ARM "))
        expected = sorted(
            [f"tum_s1_1000 dir{a}" for a in ("010", "020", "033")]
            + [f"kitti_odo_full dir{a}" for a in ("010", "020", "033")]
            + [f"arkit_s1_500 dir{a}" for a in ("010", "020", "033")]
            + [f"arkit_s1_500 const{a}" for a in ("010", "020", "033")]
            + ["scannet_s3_1000 dir010", "scannet_s3_1000 dir033", "scannet_s3_1000 const010"]
        )
        self.assertEqual(arms, expected)


class E6dQueue(unittest.TestCase):
    def test_queue_runs_the_five_registered_alphas_and_nothing_else(self):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_e6d.sh")
        env = dict(os.environ, DDD3R_DRY_RUN="1", DDD3R_PYTHON="/bin/false", DDD3R_QUEUE_LOG="/dev/null")
        done = subprocess.run(
            ["bash", script],
            capture_output=True, text=True, env=env, timeout=60,
            cwd=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stdout.splitlines() if line.startswith("TYPE=")]
        self.assertEqual(
            sorted(lines),
            sorted(f"TYPE=ddd3r_constant EXTRA=--alpha {a}" for a in ("0.10", "0.15", "0.20", "0.25", "0.33")),
        )


if __name__ == "__main__":
    unittest.main()
