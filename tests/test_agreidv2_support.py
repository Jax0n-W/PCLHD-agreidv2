from __future__ import absolute_import

import os.path as osp
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner
from agreid_identity import parse_agreidv2_manifest
from examples.PCLHD_AGReIDv2_test import modal_for_camid


def parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


def without_options(command, options):
    result = []
    index = 0
    while index < len(command):
        if command[index] in options:
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


class AGReIDV2SupportTests(unittest.TestCase):
    def test_manifest_split_and_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            inner = outer / "AG-ReID.v2"
            (inner / "query" / "0001").mkdir(parents=True)
            (inner / "gallery" / "0001").mkdir(parents=True)
            query_name = "P0001T04041A0C3F121.jpg"
            gallery_name = "P0001T04041A0C0F122.jpg"
            manifest = outer / "exp4_cctv_to_aerial.txt"
            manifest.write_text(
                "query/0001/{}\ngallery/0001/{}\n".format(
                    query_name, gallery_name), encoding="utf-8")

            query, gallery = parse_agreidv2_manifest(
                directory, "exp4", validate=False)
            self.assertEqual(len(query), 1)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(
                query[0].path,
                osp.normpath(str(inner / "query" / "0001" / query_name)))
            self.assertEqual(
                gallery[0].path,
                osp.normpath(str(
                    inner / "gallery" / "0001" / gallery_name)))
            self.assertEqual(query[0].camid, 3)
            self.assertEqual(gallery[0].camid, 0)

    def test_modal_mapping(self):
        self.assertEqual(modal_for_camid(0), 2)
        self.assertEqual(modal_for_camid(2), 1)
        self.assertEqual(modal_for_camid(3), 1)
        with self.assertRaises(ValueError):
            modal_for_camid(1)

    def test_v2_baseline_commands_only_change_dataset_and_logs(self):
        v1 = parse("--dataset-key", "agreid", "--experiment", "baseline")
        v2 = parse("--dataset-key", "agreidv2", "--experiment", "baseline")
        v1_s1, v1_s2 = runner.default_log_names(v1)
        v2_s1, v2_s2 = runner.default_log_names(v2)
        ignored = {
            "--dataset", "--dataset-key", "--stage1-log-name",
            "--stage2-log-name",
        }
        self.assertEqual(
            without_options(
                runner.build_stage1_command(v1, v1_s1, v1_s2), ignored),
            without_options(
                runner.build_stage1_command(v2, v2_s1, v2_s2), ignored))
        checkpoint = osp.join(ROOT, "synthetic_stage1.pth.tar")
        self.assertEqual(
            without_options(
                runner.build_stage2_command(
                    v1, checkpoint, v1_s1, v1_s2), ignored),
            without_options(
                runner.build_stage2_command(
                    v2, checkpoint, v2_s1, v2_s2), ignored))
        self.assertTrue(v1_s1.startswith("agreid_dt_"))
        self.assertTrue(v2_s1.startswith("agreidv2_dt_"))

    def test_v2_rejects_nonbaseline_experiments(self):
        for experiment in ("dynamic", "three_domain", "full"):
            args = parse(
                "--dataset-key", "agreidv2", "--experiment", experiment)
            with self.assertRaisesRegex(
                    ValueError, "masks and three-domain view scores"):
                runner.validate_args(args)


if __name__ == "__main__":
    unittest.main()
