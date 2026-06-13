#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DTYPE_ORDER = ["fp32", "bf16", "fp16", "fp8", "nvfp4"]
DTYPE_LABELS = {
    "fp32": "FP32",
    "bf16": "BF16",
    "fp16": "FP16",
    "fp8": "FP8",
    "nvfp4": "NVFP4",
}
DTYPE_MARKERS = {
    "fp32": "o",
    "bf16": "s",
    "fp16": "^",
    "fp8": "D",
    "nvfp4": "P",
}
GPU_MARKERS = {4: "o", 8: "s", 16: "^", 32: "D", 64: "v"}


def format_count(count: int) -> str:
    if count >= 1 << 30:
        value = count / float(1 << 30)
        return f"{value:.0f}G" if count % (1 << 30) == 0 else f"{value:.2f}G"
    if count >= 1 << 20:
        value = count / float(1 << 20)
        return f"{value:.0f}M" if count % (1 << 20) == 0 else f"{value:.2f}M"
    if count >= 1 << 10:
        value = count / float(1 << 10)
        return f"{value:.0f}K" if count % (1 << 10) == 0 else f"{value:.2f}K"
    return str(count)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def stddev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate equal-value-count dtype all-reduce CSVs and plot scaling efficiency."
    )
    parser.add_argument("csv_files", nargs="*", help="Raw CSV files to aggregate.")
    parser.add_argument("--results-dir", help="Directory that contains raw value-count scaling CSV files.")
    parser.add_argument("--run-tag", help="Run tag shared by the scaling jobs.")
    parser.add_argument("--job-ids", help="Comma-separated Slurm job IDs whose CSVs should be included.")
    parser.add_argument("--output-dir", help="Directory for outputs when --output-prefix is relative.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix, with or without a directory.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not args.csv_files and not args.results_dir:
        parser.error("pass raw CSV files or --results-dir with --run-tag/--job-ids")
    if args.results_dir and not args.run_tag and not args.job_ids:
        parser.error("--results-dir requires --run-tag or --job-ids")
    return args


def collect_files(args):
    files = set(args.csv_files)
    if args.results_dir:
        patterns = []
        if args.run_tag:
            patterns.append(os.path.join(args.results_dir, f"value_count_scaling_*g_{args.run_tag}_job*_repeat*.csv"))
        if args.job_ids:
            for job_id in [item.strip() for item in args.job_ids.split(",") if item.strip()]:
                patterns.append(os.path.join(args.results_dir, f"value_count_scaling_*g_*_job{job_id}_repeat*.csv"))
        for pattern in patterns:
            files.update(glob.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError("no CSV files matched the requested inputs")
    return files


def collect_rows(files):
    grouped = defaultdict(list)
    dtypes = set()
    ranks = set()
    counts = set()
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0] == "kind" or row[0] != "value_scaling":
                    continue
                if len(row) < 12:
                    continue
                dtype = row[1].strip().lower()
                nranks = int(row[2])
                value_count = int(row[3])
                value_label = row[4]
                buffer_bytes = int(row[5])
                scratch_bytes = int(row[6])
                avg_ms = float(row[7])
                value_rate = float(row[8])
                algbw = float(row[9])
                busbw = float(row[10])
                type_bytes = int(row[11])
                grouped[(dtype, value_count, nranks)].append(
                    {
                        "value_label": value_label,
                        "buffer_bytes": buffer_bytes,
                        "scratch_bytes": scratch_bytes,
                        "avg_ms": avg_ms,
                        "value_rate_gvals": value_rate,
                        "algbw_gbps": algbw,
                        "busbw_gbps": busbw,
                        "type_bytes": type_bytes,
                    }
                )
                dtypes.add(dtype)
                ranks.add(nranks)
                counts.add(value_count)
    if not grouped:
        raise RuntimeError("CSV files were found, but no value_scaling rows were parsed")
    dtype_order = [dtype for dtype in DTYPE_ORDER if dtype in dtypes]
    dtype_order.extend(dtype for dtype in sorted(dtypes) if dtype not in dtype_order)
    return grouped, sorted(ranks), sorted(counts), dtype_order


def aggregate(grouped, all_ranks, all_counts, all_dtypes):
    rows = []
    for dtype in all_dtypes:
        for value_count in all_counts:
            baseline = mean([sample["busbw_gbps"] for sample in grouped.get((dtype, value_count, 4), [])])
            for nranks in all_ranks:
                samples = grouped.get((dtype, value_count, nranks), [])
                if not samples:
                    continue
                avg_values = [sample["avg_ms"] for sample in samples]
                value_rates = [sample["value_rate_gvals"] for sample in samples]
                algbw_values = [sample["algbw_gbps"] for sample in samples]
                busbw_values = [sample["busbw_gbps"] for sample in samples]
                busbw_mean = mean(busbw_values)
                rows.append(
                    {
                        "dtype": dtype,
                        "dtype_label": DTYPE_LABELS.get(dtype, dtype.upper()),
                        "value_count": value_count,
                        "value_count_label": format_count(value_count),
                        "nranks": nranks,
                        "repeats": len(samples),
                        "buffer_bytes": samples[0]["buffer_bytes"],
                        "scratch_bytes": samples[0]["scratch_bytes"],
                        "type_bytes": samples[0]["type_bytes"],
                        "avg_ms_mean": mean(avg_values),
                        "avg_ms_std": stddev(avg_values),
                        "value_rate_gvals_mean": mean(value_rates),
                        "value_rate_gvals_std": stddev(value_rates),
                        "algbw_gbps_mean": mean(algbw_values),
                        "algbw_gbps_std": stddev(algbw_values),
                        "busbw_gbps_mean": busbw_mean,
                        "busbw_gbps_std": stddev(busbw_values),
                        "baseline_4g_busbw_gbps": baseline,
                        "scaling_efficiency_pct": min(100.0, 100.0 * busbw_mean / baseline) if baseline > 0 else 0.0,
                    }
                )
    return rows


def write_aggregate_csv(path, rows):
    fields = [
        "dtype", "dtype_label", "value_count", "value_count_label", "nranks", "repeats",
        "buffer_bytes", "scratch_bytes", "type_bytes", "avg_ms_mean", "avg_ms_std",
        "value_rate_gvals_mean", "value_rate_gvals_std", "algbw_gbps_mean", "algbw_gbps_std",
        "busbw_gbps_mean", "busbw_gbps_std", "baseline_4g_busbw_gbps", "scaling_efficiency_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in fields:
                if isinstance(out.get(key), float):
                    out[key] = f"{out[key]:.6f}"
            writer.writerow(out)


def make_lookup(rows):
    return {(row["dtype"], row["value_count"], row["nranks"]): row for row in rows}


def plot_by_value_count(path, rows, all_ranks, all_counts, all_dtypes, warmup, iters, repeats):
    lookup = make_lookup(rows)
    ncols = 3
    nrows = int(math.ceil(len(all_counts) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 4.2 * nrows), sharex=True, sharey=True)
    axes = list(axes.flatten() if hasattr(axes, "flatten") else [axes])
    cmap = plt.get_cmap("tab10")
    for idx, value_count in enumerate(all_counts):
        ax = axes[idx]
        for dtype_idx, dtype in enumerate(all_dtypes):
            xs = []
            ys = []
            for nranks in all_ranks:
                row = lookup.get((dtype, value_count, nranks))
                if row:
                    xs.append(nranks)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                ax.plot(xs, ys, marker=DTYPE_MARKERS.get(dtype, "o"), linewidth=2.0,
                        markersize=5.8, color=cmap(dtype_idx % 10), label=DTYPE_LABELS.get(dtype, dtype.upper()))
        ax.axhline(100.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(format_count(value_count) + " values")
        ax.set_xticks(all_ranks)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx % ncols == 0:
            ax.set_ylabel("Scaling efficiency (%)")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("GPU count")
    for ax in axes[len(all_counts):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_dtype(path, rows, all_ranks, all_counts, all_dtypes, warmup, iters, repeats):
    lookup = make_lookup(rows)
    ncols = 2
    nrows = int(math.ceil(len(all_dtypes) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.6, 4.0 * nrows), sharex=True, sharey=True)
    axes = list(axes.flatten() if hasattr(axes, "flatten") else [axes])
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(all_ranks) - 1)
    x_values = [count / float(1 << 20) for count in all_counts]
    x_labels = [format_count(count) for count in all_counts]
    for idx, dtype in enumerate(all_dtypes):
        ax = axes[idx]
        for rank_idx, nranks in enumerate(all_ranks):
            xs = []
            ys = []
            for count, x_value in zip(all_counts, x_values):
                row = lookup.get((dtype, count, nranks))
                if row:
                    xs.append(x_value)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                ax.plot(xs, ys, marker=GPU_MARKERS.get(nranks, "o"), linewidth=2.0,
                        markersize=5.5, color=cmap(rank_idx / float(denom)), label=f"{nranks} GPUs")
        ax.axhline(100.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_title(DTYPE_LABELS.get(dtype, dtype.upper()))
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx % ncols == 0:
            ax.set_ylabel("Scaling efficiency (%)")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Value count")
    for ax in axes[len(all_dtypes):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    prefix = args.output_prefix
    if args.output_dir and not os.path.isabs(prefix):
        prefix = os.path.join(args.output_dir, prefix)
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    files = collect_files(args)
    grouped, all_ranks, all_counts, all_dtypes = collect_rows(files)
    rows = aggregate(grouped, all_ranks, all_counts, all_dtypes)

    aggregate_csv = prefix + "_aggregate.csv"
    manifest = prefix + "_inputs.txt"
    by_value_png = prefix + "_by_value_count.png"
    by_value_pdf = prefix + "_by_value_count.pdf"
    by_dtype_png = prefix + "_by_dtype.png"
    by_dtype_pdf = prefix + "_by_dtype.pdf"

    write_aggregate_csv(aggregate_csv, rows)
    plot_by_value_count(by_value_png, rows, all_ranks, all_counts, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_by_value_count(by_value_pdf, rows, all_ranks, all_counts, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_by_dtype(by_dtype_png, rows, all_ranks, all_counts, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_by_dtype(by_dtype_pdf, rows, all_ranks, all_counts, all_dtypes, args.warmup, args.iters, args.repeats)
    with open(manifest, "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(path + "\n")

    print(f"wrote aggregate CSV: {aggregate_csv}")
    print(f"wrote efficiency plot by value count PNG: {by_value_png}")
    print(f"wrote efficiency plot by value count PDF: {by_value_pdf}")
    print(f"wrote efficiency plot by dtype PNG: {by_dtype_png}")
    print(f"wrote efficiency plot by dtype PDF: {by_dtype_pdf}")
    print(f"wrote input manifest: {manifest}")


if __name__ == "__main__":
    main()
