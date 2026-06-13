#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMPL_ORDER = ["nvfp4", "fp8e4m3"]
IMPL_LABELS = {
    "nvfp4": "NVFP4",
    "fp8e4m3": "FP8 E4M3",
}
IMPL_LINESTYLE = {
    "nvfp4": "-",
    "fp8e4m3": "--",
}
IMPL_MARKER = {
    "nvfp4": "o",
    "fp8e4m3": "s",
}


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        return f"{num_bytes / float(1 << 30):.0f} GiB"
    if num_bytes >= 1 << 20:
        value = num_bytes / float(1 << 20)
        return f"{value:.0f} MiB" if value.is_integer() else f"{value:.2f} MiB"
    if num_bytes >= 1 << 10:
        value = num_bytes / float(1 << 10)
        return f"{value:.0f} KiB" if value.is_integer() else f"{value:.2f} KiB"
    return f"{num_bytes} B"


def mean(values):
    return statistics.fmean(values) if values else 0.0


def stddev(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate repeated NVFP4/FP8 scaling CSVs and plot throughput.")
    parser.add_argument("--results-dir", required=True, help="Directory that contains raw scaling CSV files.")
    parser.add_argument("--run-tag", help="Run tag shared by the scaling jobs.")
    parser.add_argument("--job-ids", help="Comma-separated Slurm job IDs whose scaling CSVs should be aggregated.")
    parser.add_argument("--dtype", default="fp16", help="Logical dtype label to include in file matching and plot title.")
    parser.add_argument(
        "--metric",
        default="algbw_gbps",
        choices=("algbw_gbps", "busbw_gbps"),
        help="Throughput metric to plot.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for aggregated CSV and plot outputs, without extension.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations used in each measured run.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations averaged within each measured run.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated sweep runs per GPU count.")
    args = parser.parse_args()
    if not args.run_tag and not args.job_ids:
        parser.error("pass at least one of --run-tag or --job-ids")
    return args


def build_patterns(results_dir: str, dtype: str, run_tag: str, job_ids: str):
    patterns = []
    if run_tag:
        patterns.append(os.path.join(results_dir, f"scaling_{dtype}_*g_{run_tag}_job*_repeat*.csv"))
    if job_ids:
        for job_id in [item.strip() for item in job_ids.split(",") if item.strip()]:
            patterns.append(os.path.join(results_dir, f"scaling_{dtype}_*g_*_job{job_id}_repeat*.csv"))
            patterns.append(os.path.join(results_dir, f"scaling_{dtype}_*g_job{job_id}_repeat*.csv"))
    return patterns


def collect_rows(results_dir: str, run_tag: str, job_ids: str, dtype: str):
    patterns = build_patterns(results_dir, dtype, run_tag, job_ids)
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"no scaling CSV files matched patterns: {patterns}")

    grouped = defaultdict(lambda: defaultdict(list))
    discovered_ranks = set()
    discovered_sizes = set()
    discovered_impls = set()

    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0] == "kind":
                    continue
                if row[0] != "scaling":
                    continue
                if len(row) >= 11:
                    row_impl = row[1]
                    row_dtype = row[2]
                    nranks = int(row[3])
                    count = int(row[4])
                    logical_bytes = int(row[5])
                    buffer_bytes = int(row[6])
                    avg_ms = float(row[7])
                    algbw = float(row[8])
                    busbw = float(row[9])
                    scratch_bytes = int(row[10])
                else:
                    continue
                if row_dtype != dtype:
                    continue
                discovered_ranks.add(nranks)
                discovered_sizes.add(logical_bytes)
                discovered_impls.add(row_impl)
                grouped[(row_impl, logical_bytes)][nranks].append(
                    {
                        "count": count,
                        "buffer_bytes": buffer_bytes,
                        "avg_ms": avg_ms,
                        "algbw_gbps": algbw,
                        "busbw_gbps": busbw,
                        "scratch_bytes": scratch_bytes,
                    }
                )

    if not grouped:
        raise RuntimeError("matched files were found, but no scaling rows were parsed")
    all_impls = [impl for impl in IMPL_ORDER if impl in discovered_impls]
    for impl in sorted(discovered_impls):
        if impl not in all_impls:
            all_impls.append(impl)
    return files, grouped, sorted(discovered_ranks), sorted(discovered_sizes), all_impls


def write_aggregate_csv(output_csv: str, grouped, all_ranks, all_sizes, all_impls):
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "impl",
                "impl_label",
                "logical_bytes",
                "logical_bytes_label",
                "nranks",
                "repeats",
                "count",
                "buffer_bytes",
                "avg_ms_mean",
                "avg_ms_std",
                "algbw_gbps_mean",
                "algbw_gbps_std",
                "busbw_gbps_mean",
                "busbw_gbps_std",
                "scratch_bytes",
            ]
        )
        for logical_bytes in all_sizes:
            for impl in all_impls:
                for nranks in all_ranks:
                    samples = grouped.get((impl, logical_bytes), {}).get(nranks, [])
                    if not samples:
                        continue
                    avg_ms_values = [sample["avg_ms"] for sample in samples]
                    algbw_values = [sample["algbw_gbps"] for sample in samples]
                    busbw_values = [sample["busbw_gbps"] for sample in samples]
                    writer.writerow(
                        [
                            impl,
                            IMPL_LABELS.get(impl, impl),
                            logical_bytes,
                            format_bytes(logical_bytes),
                            nranks,
                            len(samples),
                            samples[0]["count"],
                            samples[0]["buffer_bytes"],
                            f"{mean(avg_ms_values):.6f}",
                            f"{stddev(avg_ms_values):.6f}",
                            f"{mean(algbw_values):.6f}",
                            f"{stddev(algbw_values):.6f}",
                            f"{mean(busbw_values):.6f}",
                            f"{stddev(busbw_values):.6f}",
                            samples[0]["scratch_bytes"],
                        ]
                    )


def plot_metric(output_path: str, grouped, all_ranks, all_sizes, all_impls, metric: str,
                dtype: str, warmup: int, iters: int, repeats: int):
    plt.figure(figsize=(13.0, 7.8))
    color_map = plt.get_cmap("tab10")
    for size_idx, logical_bytes in enumerate(all_sizes):
        color = color_map(size_idx % 10)
        for impl in all_impls:
            xs = []
            ys = []
            yerrs = []
            for nranks in all_ranks:
                samples = grouped.get((impl, logical_bytes), {}).get(nranks, [])
                if not samples:
                    continue
                values = [sample[metric] for sample in samples]
                xs.append(nranks)
                ys.append(mean(values))
                yerrs.append(stddev(values))
            if not xs:
                continue
            plt.errorbar(
                xs,
                ys,
                yerr=yerrs,
                marker=IMPL_MARKER.get(impl, "o"),
                linestyle=IMPL_LINESTYLE.get(impl, "-"),
                linewidth=2.0,
                markersize=6.0,
                capsize=4.0,
                color=color,
                label=f"{format_bytes(logical_bytes)} {IMPL_LABELS.get(impl, impl)}",
            )

    ylabel = "Algorithmic Throughput (GB/s, logical bytes)" if metric == "algbw_gbps" else "Bus Throughput (GB/s, logical bytes)"
    title_metric = "Algorithmic Throughput" if metric == "algbw_gbps" else "Bus Throughput"
    plt.title(
        f"NVFP4 vs FP8 AllReduce {title_metric} vs GPU Count\n"
        f"logical dtype={dtype}, shared logical element counts, warmup={warmup}, timed iters={iters}, averaged over {repeats} repeated runs"
    )
    plt.xlabel("GPU Count")
    plt.ylabel(ylabel)
    plt.xticks(all_ranks)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Logical Payload and Implementation", ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    files, grouped, all_ranks, all_sizes, all_impls = collect_rows(
        args.results_dir, args.run_tag, args.job_ids, args.dtype
    )

    aggregate_csv = args.output_prefix + "_aggregate.csv"
    plot_png = args.output_prefix + f"_{args.metric}.png"
    plot_pdf = args.output_prefix + f"_{args.metric}.pdf"
    manifest = args.output_prefix + "_inputs.txt"

    write_aggregate_csv(aggregate_csv, grouped, all_ranks, all_sizes, all_impls)
    plot_metric(plot_png, grouped, all_ranks, all_sizes, all_impls, args.metric, args.dtype,
                args.warmup, args.iters, args.repeats)
    plot_metric(plot_pdf, grouped, all_ranks, all_sizes, all_impls, args.metric, args.dtype,
                args.warmup, args.iters, args.repeats)

    with open(manifest, "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(path + "\n")

    print(f"wrote aggregate CSV: {aggregate_csv}")
    print(f"wrote plot PNG: {plot_png}")
    print(f"wrote plot PDF: {plot_pdf}")
    print(f"wrote input manifest: {manifest}")


if __name__ == "__main__":
    main()
