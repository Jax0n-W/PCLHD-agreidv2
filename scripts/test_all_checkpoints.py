#!/usr/bin/env python3
"""Evaluate the four Stage-2 final checkpoints with detailed run headers."""
from __future__ import print_function

import argparse
import os
import os.path as osp
import subprocess
import sys


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment_runner import DEFAULT_EPS, eps_all_log_suffix, eps_log_suffix

TEST_ENTRIES = {
    "agreid": osp.join(ROOT, "examples", "PCLHD_AGReID_g2ag.py"),
    "agreidv2": osp.join(ROOT, "examples", "PCLHD_AGReIDv2_test.py"),
}
EXPERIMENTS = {
    "baseline": (0, 0),
    "dynamic": (1, 0),
    "three_domain": (0, 1),
    "full": (1, 1),
}


def parser():
    p = argparse.ArgumentParser(description="Batch-test AG-ReID checkpoints")
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--dataset-key", choices=tuple(TEST_ENTRIES), default="agreid")
    p.add_argument("--test-data-dir", required=True)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument(
        "--eps-all", type=float, default=None,
        help="Stage-2 ALL DBSCAN eps; defaults to --eps")
    p.add_argument("--checkpoint-name", default="checkpoint.pth.tar")
    p.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p


def log_name(experiment, seed, dataset_key="agreid", eps=DEFAULT_EPS,
             eps_all=None):
    d_flag, t_flag = EXPERIMENTS[experiment]
    prefix = "agreidv2_dt" if dataset_key == "agreidv2" else "agreid_dt"
    return "{}_{}_d{}_t{}_channel_off_s2_seed{}{}{}".format(
        prefix, experiment, d_flag, t_flag, seed, eps_log_suffix(eps),
        eps_all_log_suffix(eps, eps_all))


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.dataset_key == "agreidv2"
            and any(name != "baseline" for name in args.experiments)):
        raise ValueError(
            "AG-ReID.v2 currently supports baseline only; D-AGVA masks and "
            "three-domain view scores for v2 are not prepared")
    logs_dir = osp.abspath(osp.expanduser(args.logs_dir))
    default_output_name = "evaluation_logs{}{}".format(
        eps_log_suffix(args.eps),
        eps_all_log_suffix(args.eps, args.eps_all))
    output_dir = args.output_dir or osp.join(logs_dir, default_output_name)
    output_dir = osp.abspath(osp.expanduser(output_dir))
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    for experiment in args.experiments:
        d_flag, t_flag = EXPERIMENTS[experiment]
        checkpoint = osp.join(
            logs_dir, log_name(
                experiment, args.seed, args.dataset_key, args.eps,
                args.eps_all),
            str(args.trial),
            args.checkpoint_name)
        note = (
            "experiment={} | D-AGVA(Stage1)={} | three-domain(Stage2)={} | "
            "channel augmentation Stage1/Stage2=OFF | checkpoint={}"
        ).format(experiment, bool(d_flag), bool(t_flag), checkpoint)
        if eps_log_suffix(args.eps):
            note += " | eps={}".format(args.eps)
        if eps_all_log_suffix(args.eps, args.eps_all):
            note += " | eps_all={}".format(args.eps_all)
        command = [
            sys.executable, TEST_ENTRIES[args.dataset_key],
            "--data-dir", osp.abspath(osp.expanduser(args.test_data_dir)),
            "--resume", checkpoint,
            "--gpu", str(args.gpu),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
        ]
        if args.dataset_key == "agreidv2":
            command += [
                "--seed", str(args.seed),
                "--output-csv", osp.join(
                    output_dir, "agreidv2_results.csv"),
            ]
        print("\n" + "=" * 88)
        print("TEST NOTE: " + note)
        print("$ " + " ".join(command))
        print("=" * 88)
        if args.dry_run:
            continue
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        log_path = osp.join(output_dir, experiment + ".log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("TEST NOTE: " + note + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        print("Saved evaluation log: {}".format(log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
