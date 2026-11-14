from __future__ import absolute_import

import contextlib
import io
import os
import os.path as osp
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from sklearn.cluster import DBSCAN


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner
import clustercontrast.utils.clustering as clustering
from clustercontrast.utils.clustering import (
    build_agreid_stage1_clusterers, build_agreid_stage2_clusterers)
from clustercontrast.utils.fast_dbscan import (
    dump_cluster_distance_if_requested, fast_dbscan_precomputed)


EPS = 0.6
MIN_SAMPLES = 4


class FastDBSCANTests(unittest.TestCase):
    def assert_matches_sklearn(self, matrix, min_samples=MIN_SAMPLES):
        before = matrix.copy()
        expected = DBSCAN(
            eps=EPS, min_samples=min_samples,
            metric="precomputed").fit_predict(matrix)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = fast_dbscan_precomputed(matrix, EPS, min_samples)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(matrix, before)
        self.assertEqual(actual.dtype, np.dtype(np.intp))

    def test_200_random_symmetric_matrices_match_exact_labels(self):
        rng = np.random.default_rng(20261114)
        sizes = list(rng.integers(5, 257, size=196))
        sizes.extend((5, 512, 1024, 2000))
        self.assertEqual(len(sizes), 200)
        sparse_values = np.array(
            [0.2] + [0.6] + [0.8] * 18, dtype=np.float64)
        dense_values = np.array(
            [0.2, 0.4, 0.6, 0.6, 0.8, 1.0], dtype=np.float64)
        with mock.patch.dict(
                os.environ, {"PCLHD_FAST_DBSCAN_BLOCK_ROWS": "64"}):
            for case_index, size in enumerate(sizes):
                dtype = np.float32 if case_index % 2 == 0 else np.float16
                values = sparse_values if case_index % 3 == 0 else dense_values
                matrix = rng.choice(values, size=(size, size)).astype(dtype)
                matrix = np.triu(matrix, 1)
                matrix += matrix.T
                np.fill_diagonal(matrix, 0)
                min_samples = 1 + (case_index % min(12, size))
                with self.subTest(
                        case=case_index, size=size, dtype=np.dtype(dtype),
                        min_samples=min_samples):
                    self.assert_matches_sklearn(matrix, min_samples)

    def test_constructed_noise_large_cluster_and_boundary_conflicts(self):
        for dtype in (np.float32, np.float16):
            all_noise = np.ones((12, 12), dtype=dtype)
            np.fill_diagonal(all_noise, 0)
            self.assert_matches_sklearn(all_noise, min_samples=2)

            one_cluster = np.full((24, 24), 0.5, dtype=dtype)
            np.fill_diagonal(one_cluster, 0)
            self.assert_matches_sklearn(one_cluster, min_samples=4)

            # Core cliques 0..3 and 4..7 are disconnected. Samples 8..11
            # are non-core border points touching a core in both components.
            boundary = np.ones((12, 12), dtype=dtype)
            np.fill_diagonal(boundary, 0)
            boundary[:4, :4] = dtype(0.2)
            boundary[4:8, 4:8] = dtype(0.2)
            np.fill_diagonal(boundary, 0)
            for border in range(8, 12):
                boundary[border, 0] = boundary[0, border] = dtype(0.5)
                boundary[border, 4] = boundary[4, border] = dtype(0.5)
            self.assert_matches_sklearn(boundary, min_samples=4)
            expected_boundary = np.array(
                [0] * 4 + [1] * 4 + [0] * 4, dtype=np.intp)
            np.testing.assert_array_equal(
                DBSCAN(
                    eps=EPS, min_samples=4,
                    metric="precomputed").fit_predict(boundary),
                expected_boundary)

            tied_components = np.ones((30, 30), dtype=dtype)
            np.fill_diagonal(tied_components, 0)
            for start in (0, 10, 20):
                tied_components[start:start + 6, start:start + 6] = dtype(0.5)
            np.fill_diagonal(tied_components, 0)
            self.assert_matches_sklearn(tied_components, min_samples=5)

    def test_backend_wrapper_and_runner_routing(self):
        fast_args = SimpleNamespace(
            eps=0.6, eps_all=0.3, cluster_n_jobs=1,
            dbscan_backend="fast")
        self.assertTrue(all(
            item.backend == "fast"
            for item in build_agreid_stage1_clusterers(fast_args).values()))
        self.assertTrue(all(
            item.backend == "fast"
            for item in build_agreid_stage2_clusterers(fast_args).values()))
        matrix = np.eye(5, dtype=np.float16)
        clusterer = build_agreid_stage1_clusterers(fast_args)["ir"]
        with mock.patch.object(
                clustering, "fast_dbscan_precomputed",
                return_value=np.zeros(5, dtype=np.intp)) as implementation:
            clusterer.fit_predict(matrix)
        self.assertIs(implementation.call_args.args[0], matrix)
        self.assertEqual(implementation.call_args.args[0].dtype, np.float16)

        args = runner.build_parser().parse_args([
            "--experiment", "baseline", "--dbscan-backend", "fast",
            "--dump-cluster-dist-epoch", "1", "--dry-run"])
        names = runner.default_log_names(args)
        stage1 = runner.build_stage1_command(args, *names)
        stage2 = runner.build_stage2_command(
            args, osp.join(ROOT, "synthetic_stage1.pth.tar"), *names)
        self.assertEqual(
            stage1[stage1.index("--dbscan-backend") + 1], "fast")
        self.assertEqual(
            stage2[stage2.index("--dbscan-backend") + 1], "fast")
        self.assertNotIn("--dump-cluster-dist-epoch", stage1)
        self.assertEqual(
            stage2[stage2.index("--dump-cluster-dist-epoch") + 1], "1")

    def test_distance_dump_is_opt_in_and_exact(self):
        matrix = np.arange(36, dtype=np.float32).reshape(6, 6)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(dump_cluster_distance_if_requested(
                matrix, epoch=0, requested_epoch=-1,
                output_dir=directory))
            self.assertIsNone(dump_cluster_distance_if_requested(
                matrix, epoch=0, requested_epoch=1,
                output_dir=directory))
            path = dump_cluster_distance_if_requested(
                matrix, epoch=1, requested_epoch=1,
                output_dir=directory)
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False), matrix)


if __name__ == "__main__":
    unittest.main()
