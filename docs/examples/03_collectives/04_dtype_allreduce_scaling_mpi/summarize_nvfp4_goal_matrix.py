#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import csv
from pathlib import Path


def read_summary(path):
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == "value_summary":
                return row
    return None


def config_from_name(path, tag):
    marker = f"_{tag}_"
    if marker not in path.name:
        return None
    tail = path.name.split(marker, 1)[1]
    if tail.startswith("job") or "_job" not in tail:
        return tag
    return tail.rsplit("_job", 1)[0]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def cap_eff(eff):
    return min(100.0, eff)


def meets_or_false(lhs, rhs):
    return False if rhs is None else lhs + 1.0e-9 >= rhs


def collect_results(results_dir, tags, expected_steps):
    data = {}
    for tag in tags:
        paths = set(results_dir.glob(f"single_case_*_{tag}_*_job*.csv"))
        paths.update(results_dir.glob(f"single_case_*_{tag}_job*.csv"))
        for path in sorted(paths):
            config = config_from_name(path, tag)
            if not config:
                continue
            row = read_summary(path)
            if row is None or int(row[8]) != expected_steps:
                continue
            label = row[4]
            nranks = int(row[2])
            key = (label, config, nranks)
            data[key] = {
                "value_count": int(row[3]),
                "steps": int(row[8]),
                "avg_ms": float(row[9]),
                "busbw": float(row[13]),
                "path": str(path),
            }
    return data


def aggregate_metrics(path, dtype):
    metrics = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dtype") == dtype:
                metrics[(row["value_count_label"], int(row["nranks"]))] = {
                    "avg_ms": float(row["avg_ms_mean"]),
                    "busbw": float(row["busbw_gbps_mean"]),
                    "eff_pct": float(row["scaling_efficiency_pct"]),
                }
    return metrics


def best_observed_4g(data):
    baselines = {}
    for (label, config, nranks), result in data.items():
        if nranks != 4:
            continue
        current = baselines.get(label)
        if current is None or result["busbw"] > current["busbw"]:
            baselines[label] = {"config": config, "busbw": result["busbw"], "path": result["path"]}
    return baselines


def summarize(data, fp8_metrics, fixed_nvfp4_metrics, ranks):
    rows = []
    observed_4g = best_observed_4g(data)
    labels = sorted({k[0] for k in data}, key=lambda label: next(v["value_count"] for k, v in data.items() if k[0] == label))
    configs = sorted({k[1] for k in data})
    for label in labels:
        for config in configs:
            baseline = data.get((label, config, 4))
            same_config_4g_busbw = baseline["busbw"] if baseline else 0.0
            best_4g = observed_4g.get(label)
            fixed_4g = fixed_nvfp4_metrics.get((label, 4))
            for rank in ranks:
                current = data.get((label, config, rank))
                if current is None:
                    continue
                same_config_eff = 100.0 * current["busbw"] / same_config_4g_busbw if same_config_4g_busbw else 0.0
                best4g_eff = 100.0 * current["busbw"] / best_4g["busbw"] if best_4g and best_4g["busbw"] else 0.0
                fixed_ch16_eff = 100.0 * current["busbw"] / fixed_4g["busbw"] if fixed_4g and fixed_4g["busbw"] else 0.0
                fp8 = fp8_metrics.get((label, rank))
                fp8_eff = None if fp8 is None else fp8["eff_pct"]
                fp8_busbw = None if fp8 is None else fp8["busbw"]
                same_config_eff_capped = cap_eff(same_config_eff)
                best4g_eff_capped = cap_eff(best4g_eff)
                fixed_ch16_eff_capped = cap_eff(fixed_ch16_eff)
                fp8_eff_capped = None if fp8_eff is None else cap_eff(fp8_eff)
                meets_fp8_busbw = meets_or_false(current["busbw"], fp8_busbw)
                fair_meets_fp8 = (
                    meets_or_false(best4g_eff_capped, fp8_eff_capped)
                    and meets_fp8_busbw
                )
                rows.append({
                    "value_count": label,
                    "config": config,
                    "nranks": rank,
                    "avg_ms": f"{current['avg_ms']:.6f}",
                    "busbw": f"{current['busbw']:.6f}",
                    "steps": current["steps"],
                    "same_config_4g_busbw": "" if baseline is None else f"{same_config_4g_busbw:.6f}",
                    "same_config_eff_pct": f"{same_config_eff:.6f}",
                    "same_config_eff_pct_capped": f"{same_config_eff_capped:.6f}",
                    "same_config_meets_fp8_eff": "" if fp8_eff_capped is None or baseline is None else str(meets_or_false(same_config_eff_capped, fp8_eff_capped)),
                    "best_observed_4g_config": "" if best_4g is None else best_4g["config"],
                    "best_observed_4g_busbw": "" if best_4g is None else f"{best_4g['busbw']:.6f}",
                    "best4g_eff_pct": f"{best4g_eff:.6f}",
                    "best4g_eff_pct_capped": f"{best4g_eff_capped:.6f}",
                    "best4g_meets_fp8_eff": "" if fp8_eff_capped is None else str(meets_or_false(best4g_eff_capped, fp8_eff_capped)),
                    "fixed_ch16_4g_busbw": "" if fixed_4g is None else f"{fixed_4g['busbw']:.6f}",
                    "fixed_ch16_eff_pct": f"{fixed_ch16_eff:.6f}",
                    "fixed_ch16_eff_pct_capped": f"{fixed_ch16_eff_capped:.6f}",
                    "fixed_ch16_meets_fp8_eff": "" if fp8_eff_capped is None else str(meets_or_false(fixed_ch16_eff_capped, fp8_eff_capped)),
                    "fp8_eff_pct": "" if fp8_eff is None else f"{fp8_eff:.6f}",
                    "fp8_eff_pct_capped": "" if fp8_eff_capped is None else f"{fp8_eff_capped:.6f}",
                    "fp8_busbw": "" if fp8_busbw is None else f"{fp8_busbw:.6f}",
                    "meets_fp8_busbw": "" if fp8_busbw is None else str(meets_fp8_busbw),
                    "fair_meets_fp8": "" if fp8_eff_capped is None or fp8_busbw is None else str(fair_meets_fp8),
                    "csv": current["path"],
                })
    return rows


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def write_csv(path, fieldnames, rows):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path.open("w", newline="")
    else:
        out = None
    try:
        writer = csv.DictWriter(out or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out:
            out.close()


def best_rows(rows, labels, ranks):
    best_by_cell = {}
    for row in rows:
        key = (row["value_count"], int(row["nranks"]))
        current_score = (
            row["fair_meets_fp8"] == "True",
            float(row["best4g_eff_pct_capped"]),
            row["meets_fp8_busbw"] == "True",
            float(row["busbw"]),
        )
        best = best_by_cell.get(key)
        best_score = None if best is None else (
            best["fair_meets_fp8"] == "True",
            float(best["best4g_eff_pct_capped"]),
            best["meets_fp8_busbw"] == "True",
            float(best["busbw"]),
        )
        if best_score is None or current_score > best_score:
            best_by_cell[key] = row

    out = []
    for label in labels:
        for rank in ranks:
            row = best_by_cell.get((label, rank))
            if row is None:
                out.append({
                    "value_count": label,
                    "nranks": rank,
                    "best_config": "",
                    "best4g_eff_pct": "",
                    "best4g_eff_pct_capped": "",
                    "same_config_eff_pct": "",
                    "same_config_eff_pct_capped": "",
                    "fixed_ch16_eff_pct": "",
                    "fixed_ch16_eff_pct_capped": "",
                    "fp8_eff_pct": "",
                    "fp8_eff_pct_capped": "",
                    "best4g_meets_fp8_eff": "False",
                    "fixed_ch16_meets_fp8_eff": "False",
                    "busbw": "",
                    "fp8_busbw": "",
                    "meets_fp8_busbw": "False",
                    "fair_meets_fp8": "False",
                    "best_observed_4g_config": "",
                    "best_observed_4g_busbw": "",
                    "same_config_4g_busbw": "",
                    "fixed_ch16_4g_busbw": "",
                    "steps": "",
                    "csv": "",
                })
            else:
                out.append({
                    "value_count": label,
                    "nranks": rank,
                    "best_config": row["config"],
                    "best4g_eff_pct": row["best4g_eff_pct"],
                    "best4g_eff_pct_capped": row["best4g_eff_pct_capped"],
                    "same_config_eff_pct": row["same_config_eff_pct"],
                    "same_config_eff_pct_capped": row["same_config_eff_pct_capped"],
                    "fixed_ch16_eff_pct": row["fixed_ch16_eff_pct"],
                    "fixed_ch16_eff_pct_capped": row["fixed_ch16_eff_pct_capped"],
                    "fp8_eff_pct": row["fp8_eff_pct"],
                    "fp8_eff_pct_capped": row["fp8_eff_pct_capped"],
                    "best4g_meets_fp8_eff": row["best4g_meets_fp8_eff"],
                    "fixed_ch16_meets_fp8_eff": row["fixed_ch16_meets_fp8_eff"],
                    "busbw": row["busbw"],
                    "fp8_busbw": row["fp8_busbw"],
                    "meets_fp8_busbw": row["meets_fp8_busbw"],
                    "fair_meets_fp8": row["fair_meets_fp8"],
                    "best_observed_4g_config": row["best_observed_4g_config"],
                    "best_observed_4g_busbw": row["best_observed_4g_busbw"],
                    "same_config_4g_busbw": row["same_config_4g_busbw"],
                    "fixed_ch16_4g_busbw": row["fixed_ch16_4g_busbw"],
                    "steps": row["steps"],
                    "csv": row["csv"],
                })
    return out


def missing_rows(best):
    return [row for row in best if row["best_config"] == "" or row["fair_meets_fp8"] != "True"]


def main():
    parser = argparse.ArgumentParser(description="Summarize tuned NVFP4 scaling efficiency across rank counts.")
    parser.add_argument("--results-dir", type=Path, default=Path("/work/09308/zhengmk/nccl/dtype_value_count_single_case_results"))
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--fp8-aggregate", type=Path, default=Path("/work/09308/zhengmk/nccl/dtype_value_count_single_case_plots/dtype_single_case_warmup5_steps50_20260605_ch16_aggregate.csv"))
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--value-counts", default="128M,256M,512M,1G")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--missing-output", type=Path)
    args = parser.parse_args()

    ranks = [int(x) for x in parse_csv_list(args.ranks)]
    labels = parse_csv_list(args.value_counts)
    rows = summarize(
        collect_results(args.results_dir, args.tag, args.expected_steps),
        aggregate_metrics(args.fp8_aggregate, "fp8"),
        aggregate_metrics(args.fp8_aggregate, "nvfp4"),
        ranks,
    )
    matrix_fieldnames = [
        "value_count", "config", "nranks",
        "avg_ms", "busbw", "steps",
        "same_config_4g_busbw", "same_config_eff_pct", "same_config_eff_pct_capped", "same_config_meets_fp8_eff",
        "best_observed_4g_config", "best_observed_4g_busbw",
        "best4g_eff_pct", "best4g_eff_pct_capped", "best4g_meets_fp8_eff",
        "fixed_ch16_4g_busbw", "fixed_ch16_eff_pct", "fixed_ch16_eff_pct_capped", "fixed_ch16_meets_fp8_eff",
        "fp8_eff_pct", "fp8_eff_pct_capped", "fp8_busbw", "meets_fp8_busbw", "fair_meets_fp8", "csv",
    ]
    write_csv(args.output, matrix_fieldnames, rows)

    if args.best_output or args.missing_output:
        best = best_rows(rows, labels, ranks)
        best_fieldnames = [
            "value_count", "nranks", "best_config",
            "best4g_eff_pct", "best4g_eff_pct_capped",
            "same_config_eff_pct", "same_config_eff_pct_capped",
            "fixed_ch16_eff_pct", "fixed_ch16_eff_pct_capped",
            "fp8_eff_pct", "fp8_eff_pct_capped",
            "best4g_meets_fp8_eff", "fixed_ch16_meets_fp8_eff",
            "busbw", "fp8_busbw", "meets_fp8_busbw", "fair_meets_fp8",
            "best_observed_4g_config", "best_observed_4g_busbw",
            "same_config_4g_busbw", "fixed_ch16_4g_busbw",
            "steps", "csv",
        ]
        if args.best_output:
            write_csv(args.best_output, best_fieldnames, best)
        if args.missing_output:
            write_csv(args.missing_output, best_fieldnames, missing_rows(best))


if __name__ == "__main__":
    main()
