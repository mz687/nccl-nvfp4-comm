#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import csv
from pathlib import Path


def read_steps(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] != "value_step":
                continue
            rows.append(row)
    return rows


def config_from_name(path, tag):
    name = path.name
    marker = f"_{tag}_"
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    return tail.rsplit("_job", 1)[0]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def collect_results(results_dir, tags):
    data = {}
    for tag in tags:
        for path in sorted(results_dir.glob(f"single_case_*_{tag}_*_job*.csv")):
            config = config_from_name(path, tag)
            if not config:
                continue
            rows = read_steps(path)
            if not rows:
                continue
            first = rows[0]
            dtype = first[1]
            nranks = int(first[2])
            value_count = int(first[3])
            label = first[4]
            key = (label, value_count, config, nranks)
            data[key] = {
                "dtype": dtype,
                "steps": len(rows),
                "avg_ms": mean([float(r[9]) for r in rows]),
                "busbw": mean([float(r[12]) for r in rows]),
                "path": str(path),
            }
    return data


def fp8_efficiencies(path):
    eff = {}
    if not path or not path.exists():
        return eff
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dtype") == "fp8" and row.get("nranks") == "64":
                eff[row["value_count_label"]] = float(row["scaling_efficiency_pct"])
    return eff


def summarize(data, fp8_eff):
    configs = sorted({k[2] for k in data})
    rows = []
    for label in sorted({k[0] for k in data}, key=lambda x: next(k[1] for k in data if k[0] == x)):
        for config in configs:
            key4 = next((k for k in data if k[0] == label and k[2] == config and k[3] == 4), None)
            key64 = next((k for k in data if k[0] == label and k[2] == config and k[3] == 64), None)
            if key4 is None or key64 is None:
                continue
            bus4 = data[key4]["busbw"]
            bus64 = data[key64]["busbw"]
            eff = 100.0 * bus64 / bus4 if bus4 else 0.0
            fp8 = fp8_eff.get(label)
            rows.append({
                "value_count": label,
                "config": config,
                "avg_ms_4g": f"{data[key4]['avg_ms']:.6f}",
                "busbw_4g": f"{bus4:.6f}",
                "steps_4g": data[key4]["steps"],
                "avg_ms_64g": f"{data[key64]['avg_ms']:.6f}",
                "busbw_64g": f"{bus64:.6f}",
                "steps_64g": data[key64]["steps"],
                "eff_pct": f"{eff:.6f}",
                "fp8_eff_pct": "" if fp8 is None else f"{fp8:.6f}",
                "meets_fp8_eff": "" if fp8 is None else str(eff >= fp8),
                "csv_4g": data[key4]["path"],
                "csv_64g": data[key64]["path"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize NVFP4 single-case channel sweep results.")
    parser.add_argument("--results-dir", type=Path, default=Path("/work/09308/zhengmk/nccl/dtype_value_count_single_case_results"))
    parser.add_argument("--tag", action="append", required=True, help="RUN_TAG to include; pass multiple times for 4G/64G tags.")
    parser.add_argument("--fp8-aggregate", type=Path, default=Path("/work/09308/zhengmk/nccl/dtype_value_count_single_case_plots/dtype_single_case_warmup5_steps50_20260605_ch16_aggregate.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = summarize(collect_results(args.results_dir, args.tag), fp8_efficiencies(args.fp8_aggregate))
    fieldnames = [
        "value_count", "config",
        "avg_ms_4g", "busbw_4g", "steps_4g",
        "avg_ms_64g", "busbw_64g", "steps_64g",
        "eff_pct", "fp8_eff_pct", "meets_fp8_eff",
        "csv_4g", "csv_64g",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = args.output.open("w", newline="")
    else:
        out = None
    try:
        writer = csv.DictWriter(out or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out:
            out.close()


if __name__ == "__main__":
    main()
