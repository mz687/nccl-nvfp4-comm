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
DTYPE_LABELS = {"fp32": "FP32", "bf16": "BF16", "fp16": "FP16", "fp8": "FP8", "nvfp4": "NVFP4"}
DTYPE_MARKERS = {"fp32": "o", "bf16": "s", "fp16": "^", "fp8": "D", "nvfp4": "P"}
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
    parser = argparse.ArgumentParser(description="Aggregate isolated single-case dtype/value-count all-reduce jobs.")
    parser.add_argument("csv_files", nargs="*", help="Raw CSV files to aggregate.")
    parser.add_argument("--results-dir", help="Directory containing raw single-case CSV files.")
    parser.add_argument("--run-tag", help="Run tag shared by the jobs.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix, with or without a directory.")
    parser.add_argument("--expected-steps", type=int, default=50)
    args = parser.parse_args()
    if not args.csv_files and not (args.results_dir and args.run_tag):
        parser.error("pass raw CSV files or --results-dir with --run-tag")
    return args


def collect_files(args):
    files = set(args.csv_files)
    if args.results_dir and args.run_tag:
        files.update(glob.glob(os.path.join(args.results_dir, f"single_case_*_{args.run_tag}_job*.csv")))
    files = sorted(files)
    if not files:
        raise FileNotFoundError("no single-case CSV files matched")
    return files


def collect_summaries(files):
    grouped = defaultdict(list)
    metadata = {}
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0] == "kind":
                    continue
                if row[0] != "value_summary":
                    continue
                dtype = row[1].strip().lower()
                nranks = int(row[2])
                value_count = int(row[3])
                key = (dtype, value_count, nranks)
                sample = {
                    "value_count_label": row[4],
                    "buffer_bytes": int(row[5]),
                    "scratch_bytes": int(row[6]),
                    "warmup_steps": int(row[7]),
                    "measured_steps": int(row[8]),
                    "avg_ms": float(row[9]),
                    "avg_ms_std": float(row[10]),
                    "value_rate_gvals": float(row[11]),
                    "algbw_gbps": float(row[12]),
                    "busbw_gbps": float(row[13]),
                    "type_bytes": int(row[14]),
                    "path": path,
                }
                grouped[key].append(sample)
                metadata.setdefault(key, sample)
    if not grouped:
        raise RuntimeError("no value_summary rows parsed")
    return grouped, metadata


def aggregate(grouped, metadata, expected_steps):
    rows = []
    for key in sorted(grouped.keys(), key=lambda k: (DTYPE_ORDER.index(k[0]) if k[0] in DTYPE_ORDER else 99, k[1], k[2])):
        dtype, value_count, nranks = key
        all_samples = sorted(grouped[key], key=lambda sample: sample["path"])
        skipped = [sample for sample in all_samples if sample["measured_steps"] != expected_steps]
        samples = [sample for sample in all_samples if sample["measured_steps"] == expected_steps]
        for sample in skipped:
            print(f"warning: {dtype} {format_count(value_count)} {nranks}g has {sample['measured_steps']} measured steps in {sample['path']}, expected {expected_steps}; skipping")
        if not samples:
            continue
        avg_ms_values = [sample["avg_ms"] for sample in samples]
        avg_ms_std_values = [sample["avg_ms_std"] for sample in samples]
        value_rate_values = [sample["value_rate_gvals"] for sample in samples]
        algbw_values = [sample["algbw_gbps"] for sample in samples]
        busbw_values = [sample["busbw_gbps"] for sample in samples]
        meta = samples[0]
        rows.append({
            "dtype": dtype,
            "dtype_label": DTYPE_LABELS.get(dtype, dtype.upper()),
            "value_count": value_count,
            "value_count_label": format_count(value_count),
            "nranks": nranks,
            "warmup_steps": meta["warmup_steps"],
            "measured_steps": meta["measured_steps"],
            "buffer_bytes": meta["buffer_bytes"],
            "scratch_bytes": meta["scratch_bytes"],
            "type_bytes": meta["type_bytes"],
            "avg_ms_mean": mean(avg_ms_values),
            "avg_ms_std": mean(avg_ms_std_values),
            "value_rate_gvals_mean": mean(value_rate_values),
            "value_rate_gvals_std": stddev(value_rate_values),
            "algbw_gbps_mean": mean(algbw_values),
            "algbw_gbps_std": stddev(algbw_values),
            "busbw_gbps_mean": mean(busbw_values),
            "busbw_gbps_std": stddev(busbw_values),
        })
    by_baseline = {(row["dtype"], row["value_count"]): row["busbw_gbps_mean"] for row in rows if row["nranks"] == 4}
    for row in rows:
        baseline = by_baseline.get((row["dtype"], row["value_count"]), 0.0)
        row["baseline_4g_busbw_gbps"] = baseline
        efficiency = 100.0 * row["busbw_gbps_mean"] / baseline if baseline > 0.0 else 0.0
        row["scaling_efficiency_pct"] = min(100.0, efficiency)
    return rows


def write_csv(path, rows):
    fields = [
        "dtype", "dtype_label", "value_count", "value_count_label", "nranks", "warmup_steps", "measured_steps",
        "buffer_bytes", "scratch_bytes", "type_bytes", "avg_ms_mean", "avg_ms_std",
        "value_rate_gvals_mean", "value_rate_gvals_std", "algbw_gbps_mean", "algbw_gbps_std",
        "busbw_gbps_mean", "busbw_gbps_std", "baseline_4g_busbw_gbps", "scaling_efficiency_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = f"{value:.6f}"
            writer.writerow(out)


def lookup(rows):
    return {(row["dtype"], row["value_count"], row["nranks"]): row for row in rows}


def plot_by_value(path, rows, ranks, counts, dtypes):
    table = lookup(rows)
    ncols = 3
    nrows = int(math.ceil(len(counts) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 4.2 * nrows), sharex=True, sharey=True)
    axes = list(axes.flatten() if hasattr(axes, "flatten") else [axes])
    cmap = plt.get_cmap("tab10")
    for idx, count in enumerate(counts):
        ax = axes[idx]
        for dtype_idx, dtype in enumerate(dtypes):
            xs, ys = [], []
            for rank in ranks:
                row = table.get((dtype, count, rank))
                if row:
                    xs.append(rank)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                ax.plot(xs, ys, marker=DTYPE_MARKERS.get(dtype, "o"), linewidth=2.0, markersize=5.8,
                        color=cmap(dtype_idx % 10), label=DTYPE_LABELS.get(dtype, dtype.upper()))
        ax.axhline(100.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(format_count(count) + " values")
        ax.set_xticks(ranks)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx % ncols == 0:
            ax.set_ylabel("Scaling efficiency (%)")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("GPU count")
    for ax in axes[len(counts):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_dtype(path, rows, ranks, counts, dtypes):
    table = lookup(rows)
    ncols = 2
    nrows = int(math.ceil(len(dtypes) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.6, 4.0 * nrows), sharex=True, sharey=True)
    axes = list(axes.flatten() if hasattr(axes, "flatten") else [axes])
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(ranks) - 1)
    x_values = [count / float(1 << 20) for count in counts]
    x_labels = [format_count(count) for count in counts]
    for idx, dtype in enumerate(dtypes):
        ax = axes[idx]
        for rank_idx, rank in enumerate(ranks):
            xs, ys = [], []
            for count, x_value in zip(counts, x_values):
                row = table.get((dtype, count, rank))
                if row:
                    xs.append(x_value)
                    ys.append(row["scaling_efficiency_pct"])
            if xs:
                ax.plot(xs, ys, marker=GPU_MARKERS.get(rank, "o"), linewidth=2.0, markersize=5.5,
                        color=cmap(rank_idx / float(denom)), label=f"{rank} GPUs")
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
    for ax in axes[len(dtypes):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    files = collect_files(args)
    grouped, metadata = collect_summaries(files)
    rows = aggregate(grouped, metadata, args.expected_steps)
    prefix = args.output_prefix
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ranks = sorted({row["nranks"] for row in rows})
    counts = sorted({row["value_count"] for row in rows})
    present_dtypes = {row["dtype"] for row in rows}
    dtypes = [dtype for dtype in DTYPE_ORDER if dtype in present_dtypes]
    dtypes.extend(dtype for dtype in sorted(present_dtypes) if dtype not in dtypes)
    write_csv(prefix + "_aggregate.csv", rows)
    plot_by_value(prefix + "_by_value_count.png", rows, ranks, counts, dtypes)
    plot_by_value(prefix + "_by_value_count.pdf", rows, ranks, counts, dtypes)
    plot_by_dtype(prefix + "_by_dtype.png", rows, ranks, counts, dtypes)
    plot_by_dtype(prefix + "_by_dtype.pdf", rows, ranks, counts, dtypes)
    with open(prefix + "_inputs.txt", "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(path + "\n")
    print(f"wrote aggregate CSV: {prefix}_aggregate.csv")
    print(f"wrote plot PNG: {prefix}_by_value_count.png")
    print(f"wrote plot PNG: {prefix}_by_dtype.png")
    print(f"input files: {len(files)}")


if __name__ == "__main__":
    main()
