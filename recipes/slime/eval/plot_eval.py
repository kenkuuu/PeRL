#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot eval results across multiple model directories.

Usage:
    python plot_eval.py [--output plot.png] [--csv-dir DIR]

Reads eval_results.json from all iter_*-hf/ directories under each MODEL_BASE,
plots avg_k and pass_k for each benchmark + overall.
同时导出与每张子图一一对应的 CSV（宽表：iter + 各模型列）。
"""

import argparse
import csv
import json
import re
from pathlib import Path
import os
import matplotlib.pyplot as plt

# ========== 配置 ==========
MODEL_BASES = [
    #"/jpfs-5p/chenyanxu.9/model/Qwen3-8B-glm5-dapo-20260411_030521",
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-muon-projected-20260409_071528",
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-muon-20260407_142933",
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-muon-projected-1-20260410_060215",
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-20260403_091551",
    # "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-sgd-20260408_152324"
]

# 从目录名提取图例标签（去掉固定前缀与末尾时间戳，避免多条曲线重名）
def short_label(path):
    name = os.path.basename(path.rstrip("/"))
    # e.g. Qwen3-8B-onpolicy-profiling-muon-projected-20260408_131541
    #   -> onpolicy-profiling-muon-projected
    # 旧写法 Qwen3-8B-(.+?)-(?:profiling|rl) 会在第一个 "-profiling" 处截断，三段路径都得到
    # "onpolicy"，dict 里后写入覆盖前写入，图上只剩一条线。
    stem = re.sub(r"^Qwen3-8B-", "", name)
    stem = re.sub(r"-\d{8}_\d{6}$", "", stem)
    return stem if stem else name


def load_results(model_base):
    """Load all eval_results.json under model_base, return {iter_num: {bmk: {avg_k, pass_k}}}."""
    results = {}
    base = Path(model_base)
    for hf_dir in sorted(base.glob("iter_*-hf")):
        result_file = hf_dir / "eval_results.json"
        if not result_file.exists():
            continue
        # extract iter number
        m = re.match(r"iter_(\d+)-hf", hf_dir.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        bmk_data = {}
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                bmk_name, metrics = row[0], row[1]
                bmk_data[bmk_name] = metrics
        results[iter_num] = bmk_data
    return results


def _safe_stem(s: str) -> str:
    """文件名安全：替换路径分隔符与特殊字符。"""
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s.strip("_") or "bmk"


def collect_series(all_data: dict, bmk: str, metric: str) -> dict[str, dict[int, float]]:
    """label -> {iter: value}，与绘图数据一致。"""
    series: dict[str, dict[int, float]] = {}
    for label, data in all_data.items():
        d: dict[int, float] = {}
        for it in sorted(data.keys()):
            row = data[it].get(bmk)
            if row is not None and metric in row:
                d[it] = row[metric]
        if d:
            series[label] = d
    return series


def write_plot_csv(path: Path, series: dict[str, dict[int, float]]) -> None:
    """宽表：第一列 iter，其余列为各模型在该 metric 上的值；缺省留空。"""
    if not series:
        return
    all_iters = sorted({it for sd in series.values() for it in sd})
    if not all_iters:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(series.keys())
    fieldnames = ["iter"] + labels
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in all_iters:
            row: dict = {"iter": it}
            for lab in labels:
                v = series[lab].get(it)
                row[lab] = v if v is not None else ""
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/jpfs/chenyanxu.9/PeRL/recipes/slime/eval/eval_results.png")
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="CSV 输出目录；默认与 --output 同目录下 <stem>_csv/",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="不写出 CSV，只画图",
    )
    args = parser.parse_args()

    out_png = Path(args.output)
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
    else:
        csv_dir = out_png.parent / f"{out_png.stem}_csv"

    # {label: {iter_num: {bmk: metrics}}}
    all_data = {}
    all_bmks = set()
    for mb in MODEL_BASES:
        label = short_label(mb)
        data = load_results(mb)
        if data:
            all_data[label] = data
            for bmk_dict in data.values():
                all_bmks.update(bmk_dict.keys())

    if not all_data:
        print("[WARN] No eval results found!")
        return

    # 排序 benchmark: 先具体 bmk，最后 overall
    bmks = sorted(b for b in all_bmks if b != "overall")
    # 每个 bmk 两张图 (avg_k, pass_k)，再加 overall 两张图
    metrics = ["avg_k", "pass_k"]

    n_bmk = len(bmks)
    n_plots = (n_bmk + 1) * 2  # +1 for overall, x2 for avg_k/pass_k
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    plot_idx = 0
    csv_written: list[Path] = []
    # Per-benchmark plots
    for bmk in bmks:
        for metric in metrics:
            ax = axes[plot_idx]
            series = collect_series(all_data, bmk, metric)
            if not args.no_csv and series:
                cpath = csv_dir / f"{_safe_stem(bmk)}__{metric}.csv"
                write_plot_csv(cpath, series)
                csv_written.append(cpath)
            for label, data in all_data.items():
                iters = sorted(data.keys())
                vals = []
                valid_iters = []
                for it in iters:
                    if bmk in data[it] and metric in data[it][bmk]:
                        valid_iters.append(it)
                        vals.append(data[it][bmk][metric])
                if valid_iters:
                    ax.plot(valid_iters, vals, marker="o", markersize=3, label=label)
            ax.set_title(f"{bmk} - {metric}")
            ax.set_xlabel("iter")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Overall plots
    for metric in metrics:
        ax = axes[plot_idx]
        series = collect_series(all_data, "overall", metric)
        if not args.no_csv and series:
            cpath = csv_dir / f"overall__{metric}.csv"
            write_plot_csv(cpath, series)
            csv_written.append(cpath)
        for label, data in all_data.items():
            iters = sorted(data.keys())
            vals = []
            valid_iters = []
            for it in iters:
                if "overall" in data[it] and metric in data[it]["overall"]:
                    valid_iters.append(it)
                    vals.append(data[it]["overall"][metric])
            if valid_iters:
                ax.plot(valid_iters, vals, marker="o", markersize=3, linewidth=2, label=label)
        ax.set_title(f"OVERALL - {metric}", fontweight="bold")
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # 隐藏多余的 subplot
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Eval Results Across Models", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved to {args.output}")
    if not args.no_csv and csv_written:
        print(f"[OK] CSV ({len(csv_written)} files) -> {csv_dir}")
        for p in csv_written:
            print(f"     {p}")


if __name__ == "__main__":
    main()
