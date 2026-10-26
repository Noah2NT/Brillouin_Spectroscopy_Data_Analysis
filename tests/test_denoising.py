"""Numerical and statistical checks; run python -m unittest discover -s tests."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from brillouin_denoising import (DenoiseConfig, reconstruct, extract_patches,
                                 reproject_patches, poisson_low_rank,
                                 poisson_deviance, thin_counts)
from simulator.generate_reconstruction_scenes import make_scene, sample_counts


class ReconstructionTests(unittest.TestCase):
    def test_patch_round_trip_including_edges(self):
        y = np.arange(6*7*9).reshape(6, 7, 9)
        for size in (1, 3, 5):
            np.testing.assert_array_equal(reproject_patches(extract_patches(y, size), y.shape, size), y)

    def test_newton_monotonic_and_positive(self):
        rng = np.random.default_rng(12)
        truth = np.exp(rng.normal(size=(40, 2))*.2 @ rng.normal(size=(2, 24)) + 1)
        y = rng.poisson(truth)
        with threadpool_limits(1):
            estimate, losses = poisson_low_rank(y, rank=3, iterations=30)
        self.assertTrue(np.all(np.isfinite(estimate)))
        self.assertTrue(np.all(estimate > 0))
        self.assertTrue(np.all(np.diff(losses) <= 1e-7))
        self.assertLess(np.mean((estimate-truth)**2), np.mean((y-truth)**2))

    def test_zero_cube_and_reproducibility(self):
        with threadpool_limits(1):
            np.testing.assert_array_equal(reconstruct(np.zeros((5, 5, 12))), 0)
            y = np.random.default_rng(4).poisson(3, (6, 7, 12))
            c = DenoiseConfig(iterations=4)
            np.testing.assert_array_equal(reconstruct(y, c), reconstruct(y, c))

    def test_trace_matches_uninstrumented_fit_and_overlap(self):
        y = np.random.default_rng(9).poisson(4, (7, 8, 16))
        config = DenoiseConfig(iterations=3, patch=3, clusters=2)
        trace = {}
        with threadpool_limits(1):
            expected = reconstruct(y, config)
            actual = reconstruct(y, config, trace=trace)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(trace["cubes"][-1], actual, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(reproject_patches(trace["final_patches"], y.shape, 3), actual)
        self.assertEqual(len(trace["names"]), len(trace["cubes"]))
        self.assertTrue(np.all(np.diff(trace["losses"]) <= 1e-7))
        for cube in trace["cubes"]:
            self.assertEqual(cube.shape, y.shape)
            self.assertTrue(np.all(np.isfinite(cube)))
            self.assertTrue(np.all(cube >= 0))

    def test_zero_trace_has_initial_prediction(self):
        trace = {}
        with threadpool_limits(1):
            output = reconstruct(np.zeros((5, 5, 8)), trace=trace)
        self.assertEqual(trace["names"], ["initial"])
        np.testing.assert_array_equal(trace["cubes"][0], output)

    def test_real_data_scores_are_changes_not_truth_errors(self):
        from analysis.patch_walkthrough import score
        raw = np.ones((3, 4, 8))*10
        frequency = np.linspace(3, 8, 8)
        mask = np.ones((3, 4), bool)
        same = score(raw, None, frequency, None, mask, raw)
        self.assertEqual(same['shape_change_pct'], 0)
        self.assertEqual(same['residual_rms'], 0)
        doubled = score(raw*2, None, frequency, None, mask, raw)
        self.assertEqual(doubled['shape_change_pct'], 0)
        self.assertEqual(doubled['total_intensity_change_pct'], 100)
        self.assertEqual(doubled['residual_rms'], 10)
        self.assertNotIn('shape_nrmse', doubled)
        self.assertNotIn('independent_deviance_per_bin', doubled)

    def test_thinning_poisson_moments(self):
        y = np.random.default_rng(3).poisson(12, (100, 100, 20))
        a, b = thin_counts(y, .25, 8)
        np.testing.assert_array_equal(a+b, y)
        self.assertAlmostEqual(a.mean(), 3, delta=.025)
        self.assertAlmostEqual(a.var(), 3, delta=.04)
        self.assertAlmostEqual(np.cov(a.ravel(), b.ravel())[0, 1], 0, delta=.05)

    def test_independent_residual_calibration(self):
        rng = np.random.default_rng(3)
        mean = np.full((70, 80, 32), 5.)
        estimate = mean + .5
        y = rng.poisson(mean)
        actual = np.mean((y-estimate)**2/mean)
        self.assertAlmostEqual(actual, 1+.5**2/5, delta=.02)
        np.testing.assert_array_equal(poisson_deviance([0, 4], [0, 4]), [2e-12, 0])

    def test_simulator_shot_noise_and_fixed_truth(self):
        scene = make_scene("hydrogel_gradient", size=12, bins=32)
        y = sample_counts(scene["rate"], .25, 12)
        np.testing.assert_array_equal(y, sample_counts(scene["rate"], .25, 12))
        self.assertTrue(np.all(scene["rate"] > 0))
        self.assertEqual(y.shape, (12, 12, 32))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            reconstruct(np.array([[1, 2]]))
        with self.assertRaises(ValueError):
            thin_counts(np.full((2, 2, 3), .5), .5)
        with self.assertRaises(ValueError):
            reconstruct(np.full((2, 2, 3), -1))

    def test_stable_but_erased_features_fail_metrics(self):
        from analysis.reconstruction_benchmark import metrics
        scene = make_scene("cell_domains", size=16, bins=64)
        truth = scene["rate"]
        erased = np.broadcast_to(truth.mean(axis=(0, 1)), truth.shape).copy()
        score = metrics(erased, scene, 1., sample_counts(truth, 1., 31), erased)
        self.assertAlmostEqual(score["high_exposure_shape_delta"], 0)
        self.assertAlmostEqual(score["contrast_ratio"], 0)
        self.assertGreater(score["centroid_rmse_mhz"], 15)

    def test_csv_loader_sorts_coordinates_and_rejects_duplicates(self):
        import pandas as pd
        from analysis.reconstruct_scan import load_input
        rows = [dict(X=x, Y=y, startFrequency=3, endFrequency=8,
                     **{"Raw Data":",".join([str(10*y+x)]*6)})
                for x, y in [(1, 1), (0, 0), (1, 0), (0, 1)]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            cube, frequency, x, y = load_input(path, (3, 8))
            np.testing.assert_array_equal(cube[:, :, 0], [[0, 1], [10, 11]])
            pd.DataFrame(rows+[rows[0]]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_input(path, (3, 8))


if __name__ == "__main__":
    unittest.main()
