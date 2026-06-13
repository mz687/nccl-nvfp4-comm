#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
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


DTYPE_ORDER = ["fp32", "bf16", "fp16", "fp8"]
DTYPE_LABELS = {
    "fp32": "FP32",
    "bf16": "BF16",
    "fp16": "FP16",
    "fp8": "FP8",
}
DTYPE_MARKERS = {
    "fp32": "o",
    "bf16": "s",
    "fp16": "^",
    "fp8": "D",
}
GPU_MARKERS = {
    4: "o",
    8: "s",
    16: "^",
    32: "D",
    64: "v",
}


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        value = num_bytes / float(1 << 30)
        return f"{value:.0f} GiB" if value.is_integer() else f"{value:.2f} GiB"
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
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate dtype all-reduce scaling CSVs and plot scaling efficiency. "
            "Efficiency is normalized to the 4-GPU bus bandwidth for each dtype and message size."
        )
    )
    parser.add_argument("csv_files", nargs="*", help="Raw CSV files to aggregate.")
    parser.add_argument("--results-dir", help="Directory that contains raw scaling CSV files.")
    parser.add_argument("--run-tag", help="Run tag shared by the scaling jobs.")
    parser.add_argument("--job-ids", help="Comma-separated Slurm job IDs whose CSVs should be included.")
    parser.add_argument("--output-dir", help="Directory for plot and aggregate outputs.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix, with or without a directory.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations used in each measured run.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations averaged within each measured run.")
    parser.add_argument("--repeats", type=int, default=3, help="Expected repeated sweep runs per GPU count.")
    args = parser.parse_args()
    if not args.csv_files and not args.results_dir:
        parser.error("pass raw CSV files or --results-dir with --run-tag/--job-ids")
    if args.results_dir and not args.run_tag and not args.job_ids:
        parser.error("--results-dir requires --run-tag or --job-ids")
    return args


def build_patterns(results_dir: str, run_tag: str, job_ids: str):
    patterns = []
    if run_tag:
        patterns.append(os.path.join(results_dir, f"dtype_scaling_*g_{run_tag}_job*_repeat*.csv"))
    if job_ids:
        for job_id in [item.strip() for item in job_ids.split(",") if item.strip()]:
            patterns.append(os.path.join(results_dir, f"dtype_scaling_*g_*_job{job_id}_repeat*.csv"))
    return patterns


def collect_files(args):
    files = set(args.csv_files)
    if args.results_dir:
        for pattern in build_patterns(args.results_dir, args.run_tag, args.job_ids):
            files.update(glob.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError("no CSV files matched the requested inputs")
    return files


def collect_rows(files):
    grouped = defaultdict(list)
    ranks = set()
    sizes = set()
    dtypes = set()

    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0] == "kind" or row[0] != "scaling":
                    continue
                if len(row) < 9:
                    continue
                dtype = row[1].strip().lower()
                nranks = int(row[2])
                count = int(row[3])
                message_bytes = int(row[4])
                buffer_bytes = int(row[5])
                avg_ms = float(row[6])
                algbw = float(row[7])
                busbw = float(row[8])
                key = (dtype, message_bytes, nranks)
                grouped[key].append(
                    {
                        "count": count,
                        "buffer_bytes": buffer_bytes,
                        "avg_ms": avg_ms,
                        "algbw_gbps": algbw,
                        "busbw_gbps": busbw,
                    }
                )
                ranks.add(nranks)
                sizes.add(message_bytes)
                dtypes.add(dtype)

    if not grouped:
        raise RuntimeError("CSV files were found, but no scaling rows were parsed")
    dtype_order = [dtype for dtype in DTYPE_ORDER if dtype in dtypes]
    dtype_order.extend(dtype for dtype in sorted(dtypes) if dtype not in dtype_order)
    return grouped, sorted(ranks), sorted(sizes), dtype_order


def aggregate(grouped, all_ranks, all_sizes, all_dtypes):
    rows = []
    for dtype in all_dtypes:
        for message_bytes in all_sizes:
            baseline_samples = grouped.get((dtype, message_bytes, 4), [])
            baseline_busbw = mean([sample["busbw_gbps"] for sample in baseline_samples])
            for nranks in all_ranks:
                samples = grouped.get((dtype, message_bytes, nranks), [])
                if not samples:
                    continue
                avg_values = [sample["avg_ms"] for sample in samples]
                algbw_values = [sample["algbw_gbps"] for sample in samples]
                busbw_values = [sample["busbw_gbps"] for sample in samples]
                busbw_mean = mean(busbw_values)
                raw_efficiency = 100.0 * busbw_mean / baseline_busbw if baseline_busbw > 0.0 else 0.0
                efficiency = min(100.0, raw_efficiency)
                rows.append(
                    {
                        "dtype": dtype,
                        "dtype_label": DTYPE_LABELS.get(dtype, dtype.upper()),
                        "message_bytes": message_bytes,
                        "message_label": format_bytes(message_bytes),
                        "message_mib": message_bytes / float(1 << 20),
                        "nranks": nranks,
                        "repeats": len(samples),
                        "count": samples[0]["count"],
                        "buffer_bytes": samples[0]["buffer_bytes"],
                        "avg_ms_mean": mean(avg_values),
                        "avg_ms_std": stddev(avg_values),
                        "algbw_gbps_mean": mean(algbw_values),
                        "algbw_gbps_std": stddev(algbw_values),
                        "busbw_gbps_mean": busbw_mean,
                        "busbw_gbps_std": stddev(busbw_values),
                        "baseline_4g_busbw_gbps": baseline_busbw,
                        "scaling_efficiency_pct": efficiency,
                    }
                )
    return rows


def write_aggregate_csv(output_csv, rows):
    fieldnames = [
        "dtype",
        "dtype_label",
        "message_bytes",
        "message_label",
        "message_mib",
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
        "baseline_4g_busbw_gbps",
        "scaling_efficiency_pct",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in [
                "message_mib",
                "avg_ms_mean",
                "avg_ms_std",
                "algbw_gbps_mean",
                "algbw_gbps_std",
                "busbw_gbps_mean",
                "busbw_gbps_std",
                "baseline_4g_busbw_gbps",
                "scaling_efficiency_pct",
            ]:
                out[key] = f"{out[key]:.6f}"
            writer.writerow(out)


def row_lookup(rows):
    return {(row["dtype"], row["message_bytes"], row["nranks"]): row for row in rows}


def plot_efficiency_by_size(output_path, rows, all_ranks, all_sizes, all_dtypes, warmup, iters, repeats):
    lookup = row_lookup(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()
    color_map = plt.get_cmap("tab10")
    for size_idx, message_bytes in enumerate(all_sizes):
        ax = axes[size_idx]
        for dtype_idx, dtype in enumerate(all_dtypes):
            xs = []
            ys = []
            for nranks in all_ranks:
                row = lookup.get((dtype, message_bytes, nranks))
                if row:
                    xs.append(nranks)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker=DTYPE_MARKERS.get(dtype, "o"),
                    linewidth=2.2,
                    markersize=6.0,
                    color=color_map(dtype_idx % 10),
                    label=DTYPE_LABELS.get(dtype, dtype.upper()),
                )
        ax.axhline(100.0, color="#808080", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(format_bytes(message_bytes))
        ax.set_xticks(all_ranks)
        ax.grid(True, linestyle="--", alpha=0.35)
        if size_idx % 2 == 0:
            ax.set_ylabel("Scaling efficiency (%)")
        if size_idx >= 2:
            ax.set_xlabel("GPU count")
    for ax in axes[len(all_sizes):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_efficiency_by_dtype(output_path, rows, all_ranks, all_sizes, all_dtypes, warmup, iters, repeats):
    lookup = row_lookup(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True, sharey=True)
    axes = axes.flatten()
    rank_color_map = plt.get_cmap("viridis")
    rank_positions = {rank: idx for idx, rank in enumerate(all_ranks)}
    denom = max(1, len(all_ranks) - 1)
    x_values = [size / float(1 << 20) for size in all_sizes]
    x_labels = [format_bytes(size) for size in all_sizes]

    for dtype_idx, dtype in enumerate(all_dtypes):
        ax = axes[dtype_idx]
        for nranks in all_ranks:
            ys = []
            xs = []
            for message_bytes, x_value in zip(all_sizes, x_values):
                row = lookup.get((dtype, message_bytes, nranks))
                if row:
                    xs.append(x_value)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                color = rank_color_map(rank_positions[nranks] / float(denom))
                ax.plot(
                    xs,
                    ys,
                    marker=GPU_MARKERS.get(nranks, "o"),
                    linewidth=2.0,
                    markersize=5.5,
                    color=color,
                    label=f"{nranks} GPUs",
                )
        ax.axhline(100.0, color="#808080", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_title(DTYPE_LABELS.get(dtype, dtype.upper()))
        ax.grid(True, linestyle="--", alpha=0.35)
        if dtype_idx % 2 == 0:
            ax.set_ylabel("Scaling efficiency (%)")
        if dtype_idx >= 2:
            ax.set_xlabel("Message size")
    for ax in axes[len(all_dtypes):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    prefix = args.output_prefix
    if args.output_dir and not os.path.isabs(prefix):
        prefix = os.path.join(args.output_dir, prefix)
    output_dir = os.path.dirname(prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    files = collect_files(args)
    grouped, all_ranks, all_sizes, all_dtypes = collect_rows(files)
    rows = aggregate(grouped, all_ranks, all_sizes, all_dtypes)

    aggregate_csv = prefix + "_aggregate.csv"
    manifest = prefix + "_inputs.txt"
    by_size_png = prefix + "_by_message_size.png"
    by_size_pdf = prefix + "_by_message_size.pdf"
    by_dtype_png = prefix + "_by_dtype.png"
    by_dtype_pdf = prefix + "_by_dtype.pdf"

    write_aggregate_csv(aggregate_csv, rows)
    plot_efficiency_by_size(by_size_png, rows, all_ranks, all_sizes, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_efficiency_by_size(by_size_pdf, rows, all_ranks, all_sizes, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_efficiency_by_dtype(by_dtype_png, rows, all_ranks, all_sizes, all_dtypes, args.warmup, args.iters, args.repeats)
    plot_efficiency_by_dtype(by_dtype_pdf, rows, all_ranks, all_sizes, all_dtypes, args.warmup, args.iters, args.repeats)

    with open(manifest, "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(path + "\n")

    print(f"wrote aggregate CSV: {aggregate_csv}")
    print(f"wrote efficiency plot by message size PNG: {by_size_png}")
    print(f"wrote efficiency plot by message size PDF: {by_size_pdf}")
    print(f"wrote efficiency plot by dtype PNG: {by_dtype_png}")
    print(f"wrote efficiency plot by dtype PDF: {by_dtype_pdf}")
    print(f"wrote input manifest: {manifest}")


if __name__ == "__main__":
    main()
