from __future__ import absolute_import

import os.path as osp
import sys
import unittest
from types import SimpleNamespace


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner
from clustercontrast.utils.clustering import (
    append_eps_log_suffix,
    build_agreid_stage1_clusterers,
    build_agreid_stage2_clusterers,
    build_agreid_three_domain_clusterers,
)


class ClusteringEpsTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(eps=0.6, cluster_n_jobs=1)

    def test_stage1_dbscan_uses_requested_eps(self):
        clusterers = build_agreid_stage1_clusterers(self.args)
        self.assertEqual(set(clusterers), {"ir", "rgb"})
        self.assertTrue(all(item.eps == 0.6 for item in clusterers.values()))

    def test_stage2_dbscan_uses_requested_eps(self):
        clusterers = build_agreid_stage2_clusterers(self.args)
        self.assertEqual(set(clusterers), {"ir", "rgb", "all"})
        self.assertTrue(all(item.eps == 0.6 for item in clusterers.values()))

    def test_three_domain_dbscan_uses_requested_eps(self):
        clusterers = build_agreid_three_domain_clusterers(self.args)
        self.assertEqual(set(clusterers), {"far", "near", "ground", "all"})
        self.assertTrue(all(item.eps == 0.6 for item in clusterers.values()))

    def test_runner_forwards_eps_to_both_stages(self):
        args = runner.build_parser().parse_args([
            "--experiment", "baseline", "--eps", "0.6", "--dry-run"])
        stage1_name, stage2_name = runner.default_log_names(args)
        stage1 = runner.build_stage1_command(
            args, stage1_name, stage2_name)
        stage2 = runner.build_stage2_command(
            args, osp.join(ROOT, "synthetic_stage1.pth.tar"),
            stage1_name, stage2_name)
        self.assertEqual(stage1[stage1.index("--eps") + 1], "0.6")
        self.assertEqual(stage2[stage2.index("--eps") + 1], "0.6")

    def test_nondefault_eps_isolated_in_log_names(self):
        default_args = runner.build_parser().parse_args([
            "--experiment", "baseline", "--dry-run"])
        tuned_args = runner.build_parser().parse_args([
            "--experiment", "baseline", "--eps", "0.6", "--dry-run"])
        default_names = runner.default_log_names(default_args)
        tuned_names = runner.default_log_names(tuned_args)
        self.assertTrue(all("_eps" not in name for name in default_names))
        self.assertTrue(all(name.endswith("_eps0.6") for name in tuned_names))
        self.assertEqual(
            append_eps_log_suffix("direct_engine", 0.6),
            "direct_engine_eps0.6")
        self.assertEqual(
            append_eps_log_suffix("direct_engine_eps0.6", 0.6),
            "direct_engine_eps0.6")

    def test_internal_engine_defaults_are_point_three(self):
        with open(osp.join(ROOT, "_train_agreid_engine.py"),
                  "r", encoding="utf-8") as handle:
            baseline_source = handle.read()
        with open(osp.join(ROOT, "_train_agreid_three_domain_engine.py"),
                  "r", encoding="utf-8") as handle:
            three_domain_source = handle.read()
        self.assertIn(
            "parser.add_argument('--eps', type=float, default=0.3",
            baseline_source)
        self.assertIn(
            'parser.add_argument("--eps", type=float, default=0.3)',
            three_domain_source)


if __name__ == "__main__":
    unittest.main()
