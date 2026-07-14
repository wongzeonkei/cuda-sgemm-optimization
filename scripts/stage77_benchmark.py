#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGE75_EXE = PROJECT_ROOT / "bin" / "sgemm_stage75_vectorized"
STAGE76_EXE = PROJECT_ROOT / "bin" / "sgemm_stage76_double_buffer"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "stage77_runs"
SUMMARY_CSV = PROJECT_ROOT / "results" / "stage77_benchmark_summary.csv"
SUMMARY_MD = PROJECT_ROOT / "docs" / "stage77_benchmark_results.md"
ENVIRONMENT_TXT = PROJECT_ROOT / "results" / "stage77_environment.txt"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    m: int
    n: int
    k: int
    warmup: int
    iterations: int


@dataclass
class Measurement:
    case_name: str
    source: str
    implementation: str
    repeat: int
    m: int
    n: int
    k: int
    latency_ms: float
    gflops: float

    @property
    def full_name(self) -> str:
        return f"{self.source}:{self.implementation}"


DEFAULT_CASES = [
    BenchmarkCase("square_512", 512, 512, 512, 10, 100),
    BenchmarkCase("square_1024", 1024, 1024, 1024, 10, 80),
    BenchmarkCase("square_1536", 1536, 1536, 1536, 10, 50),
    BenchmarkCase("square_2048", 2048, 2048, 2048, 10, 40),
    BenchmarkCase("square_3072", 3072, 3072, 3072, 5, 25),
    BenchmarkCase("square_4096", 4096, 4096, 4096, 5, 20),
    BenchmarkCase("aligned_edge", 1000, 1032, 780, 10, 60),
    BenchmarkCase("scalar_fallback", 1000, 1030, 777, 10, 60),
]

QUICK_CASES = [
    BenchmarkCase("square_512", 512, 512, 512, 2, 5),
    BenchmarkCase("square_1024", 1024, 1024, 1024, 2, 5),
]


def run_text_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return f"Command not found: {command[0]}"


def collect_environment() -> None:
    ENVIRONMENT_TXT.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        ("Date", run_text_command(["date", "-Iseconds"])),
        ("Git commit", run_text_command(["git", "rev-parse", "HEAD"])),
        ("Git status", run_text_command(["git", "status", "-sb"])),
        ("NVCC", run_text_command(["nvcc", "--version"])),
        (
            "GPU",
            run_text_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,"
                    "temperature.gpu,clocks.sm,clocks.mem,"
                    "power.limit",
                    "--format=csv",
                ]
            ),
        ),
    ]

    with ENVIRONMENT_TXT.open("w", encoding="utf-8") as file:
        for title, content in sections:
            file.write(f"===== {title} =====\n")
            file.write(content)
            file.write("\n\n")


def find_numeric_value(
    row: dict[str, str],
    candidate_keys: Iterable[str],
) -> float | None:
    for key in candidate_keys:
        value = row.get(key)

        if value is None or value == "":
            continue

        try:
            return float(value)
        except ValueError:
            continue

    return None


def parse_result_csv(
    csv_path: Path,
    source: str,
    case: BenchmarkCase,
    repeat: int,
) -> list[Measurement]:
    measurements: list[Measurement] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(f"No CSV header found in {csv_path}")

        for index, row in enumerate(reader):
            implementation = (
                row.get("implementation")
                or row.get("kernel")
                or row.get("name")
                or f"row_{index}"
            )

            latency_ms = find_numeric_value(
                row,
                [
                    "mean_latency_ms",
                    "latency_ms",
                    "mean_ms",
                    "time_ms",
                ],
            )

            gflops = find_numeric_value(
                row,
                [
                    "gflops",
                    "performance_gflops",
                    "mean_gflops",
                ],
            )

            if latency_ms is None:
                continue

            if latency_ms <= 0.0:
                continue

            if gflops is None:
                operations = (
                    2.0
                    * float(case.m)
                    * float(case.n)
                    * float(case.k)
                )

                gflops = operations / (latency_ms / 1000.0) / 1e9

            measurements.append(
                Measurement(
                    case_name=case.name,
                    source=source,
                    implementation=implementation.strip(),
                    repeat=repeat,
                    m=case.m,
                    n=case.n,
                    k=case.k,
                    latency_ms=latency_ms,
                    gflops=gflops,
                )
            )

    if not measurements:
        raise RuntimeError(
            f"No benchmark measurements parsed from {csv_path}"
        )

    return measurements


def run_one_program(
    executable: Path,
    source: str,
    case: BenchmarkCase,
    repeat: int,
) -> list[Measurement]:
    case_directory = OUTPUT_ROOT / case.name
    case_directory.mkdir(parents=True, exist_ok=True)

    csv_path = (
        case_directory
        / f"{source}_repeat_{repeat:02d}.csv"
    )

    log_path = (
        case_directory
        / f"{source}_repeat_{repeat:02d}.txt"
    )

    command = [
        str(executable),
        str(case.m),
        str(case.n),
        str(case.k),
        str(case.warmup),
        str(case.iterations),
        str(csv_path),
    ]

    print(
        f"[RUN] {case.name:<18} "
        f"{source:<8} "
        f"repeat={repeat}"
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log_path.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(
            f"Benchmark failed: {' '.join(command)}"
        )

    return parse_result_csv(
        csv_path=csv_path,
        source=source,
        case=case,
        repeat=repeat,
    )


def is_cublas(name: str) -> bool:
    return "cublas" in name.lower()


def median(values: list[float]) -> float:
    return statistics.median(values)


def sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def write_raw_csv(measurements: list[Measurement]) -> None:
    raw_path = PROJECT_ROOT / "results" / "stage77_benchmark_raw.csv"

    with raw_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "case",
                "source",
                "implementation",
                "repeat",
                "M",
                "N",
                "K",
                "latency_ms",
                "gflops",
            ]
        )

        for item in measurements:
            writer.writerow(
                [
                    item.case_name,
                    item.source,
                    item.implementation,
                    item.repeat,
                    item.m,
                    item.n,
                    item.k,
                    f"{item.latency_ms:.9f}",
                    f"{item.gflops:.6f}",
                ]
            )


def aggregate_measurements(
    measurements: list[Measurement],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str],
        list[Measurement],
    ] = defaultdict(list)

    for item in measurements:
        grouped[(item.case_name, item.full_name)].append(item)

    aggregate_rows: list[dict[str, object]] = []

    for (case_name, full_name), items in sorted(grouped.items()):
        latency_values = [item.latency_ms for item in items]
        gflops_values = [item.gflops for item in items]

        first = items[0]

        median_latency = median(latency_values)
        median_gflops = median(gflops_values)
        stdev_gflops = sample_stdev(gflops_values)

        coefficient_of_variation = (
            stdev_gflops / median_gflops * 100.0
            if median_gflops > 0.0
            else 0.0
        )

        aggregate_rows.append(
            {
                "case": case_name,
                "source": first.source,
                "implementation": first.implementation,
                "full_name": full_name,
                "M": first.m,
                "N": first.n,
                "K": first.k,
                "runs": len(items),
                "median_latency_ms": median_latency,
                "min_latency_ms": min(latency_values),
                "max_latency_ms": max(latency_values),
                "median_gflops": median_gflops,
                "min_gflops": min(gflops_values),
                "max_gflops": max(gflops_values),
                "stdev_gflops": stdev_gflops,
                "coefficient_of_variation_percent":
                    coefficient_of_variation,
                "cublas_ratio_percent": 0.0,
            }
        )

    rows_by_case_source: dict[
        tuple[str, str],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in aggregate_rows:
        rows_by_case_source[
            (str(row["case"]), str(row["source"]))
        ].append(row)

    for rows in rows_by_case_source.values():
        cublas_rows = [
            row
            for row in rows
            if is_cublas(str(row["implementation"]))
        ]

        if not cublas_rows:
            continue

        cublas_gflops = max(
            float(row["median_gflops"])
            for row in cublas_rows
        )

        for row in rows:
            row["cublas_ratio_percent"] = (
                float(row["median_gflops"])
                / cublas_gflops
                * 100.0
                if cublas_gflops > 0.0
                else 0.0
            )

    return aggregate_rows


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "source",
        "implementation",
        "M",
        "N",
        "K",
        "runs",
        "median_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "median_gflops",
        "min_gflops",
        "max_gflops",
        "stdev_gflops",
        "coefficient_of_variation_percent",
        "cublas_ratio_percent",
    ]

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            output_row = {
                key: row[key]
                for key in fieldnames
            }

            for key in [
                "median_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "median_gflops",
                "min_gflops",
                "max_gflops",
                "stdev_gflops",
                "coefficient_of_variation_percent",
                "cublas_ratio_percent",
            ]:
                output_row[key] = f"{float(output_row[key]):.6f}"

            writer.writerow(output_row)


def choose_best_custom_kernel(
    case_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    custom_rows = [
        row
        for row in case_rows
        if not is_cublas(str(row["implementation"]))
    ]

    if not custom_rows:
        return None

    return max(
        custom_rows,
        key=lambda row: float(row["median_gflops"]),
    )


def write_summary_markdown(
    rows: list[dict[str, object]],
    repeats: int,
) -> None:
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)

    rows_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        rows_by_case[str(row["case"])].append(row)

    lines: list[str] = []

    lines.extend(
        [
            "# Stage 7.7 Unified Benchmark Results",
            "",
            "## Protocol",
            "",
            f"- Repeats per executable: {repeats}",
            "- Statistic used for dispatch decisions: median",
            "- GPU timing: CUDA Events inside each executable",
            "- Reference: cuBLAS Pedantic FP32",
            "- Execution order alternates between Stage 7.5 and Stage 7.6",
            "- Raw logs and CSV files are stored under `results/stage77_runs/`",
            "",
        ]
    )

    for case_name in sorted(rows_by_case):
        case_rows = rows_by_case[case_name]

        first = case_rows[0]

        lines.extend(
            [
                f"## {case_name}",
                "",
                (
                    f"Shape: {first['M']} x "
                    f"{first['N']} x {first['K']}"
                ),
                "",
                (
                    "| Source | Implementation | Runs | "
                    "Median ms | Median GFLOPS | "
                    "Min GFLOPS | Max GFLOPS | "
                    "CV | cuBLAS ratio |"
                ),
                (
                    "|---|---|---:|---:|---:|---:|"
                    "---:|---:|---:|"
                ),
            ]
        )

        sorted_rows = sorted(
            case_rows,
            key=lambda row: float(row["median_gflops"]),
            reverse=True,
        )

        for row in sorted_rows:
            lines.append(
                "| "
                f"{row['source']} | "
                f"{row['implementation']} | "
                f"{row['runs']} | "
                f"{float(row['median_latency_ms']):.6f} | "
                f"{float(row['median_gflops']):.3f} | "
                f"{float(row['min_gflops']):.3f} | "
                f"{float(row['max_gflops']):.3f} | "
                f"{float(row['coefficient_of_variation_percent']):.3f}% | "
                f"{float(row['cublas_ratio_percent']):.3f}% |"
            )

        best = choose_best_custom_kernel(case_rows)

        lines.append("")

        if best is not None:
            lines.append(
                "**Best custom implementation:** "
                f"`{best['full_name']}` at "
                f"{float(best['median_gflops']):.3f} GFLOPS."
            )
            lines.append("")

    lines.extend(
        [
            "## Dispatch Decision",
            "",
            "The final dispatch policy must be selected from median results,",
            "not from a single historical run. The scalar fallback and",
            "float4-aligned paths should be evaluated separately.",
            "",
        ]
    )

    SUMMARY_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated Stage 7.5/7.6 CUDA SGEMM benchmark."
        )
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions for each executable and case.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Cooling delay in seconds between process launches.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke test.",
    )

    parser.add_argument(
        "--skip-stage75",
        action="store_true",
        help="Do not execute the Stage 7.5 binary.",
    )

    parser.add_argument(
        "--skip-stage76",
        action="store_true",
        help="Do not execute the Stage 7.6 binary.",
    )

    return parser.parse_args()


def validate_executable(
    executable: Path,
    skipped: bool,
) -> None:
    if skipped:
        return

    if not executable.exists():
        raise FileNotFoundError(
            f"Executable not found: {executable}"
        )

    if not os.access(executable, os.X_OK):
        raise PermissionError(
            f"Executable is not executable: {executable}"
        )


def main() -> int:
    args = parse_arguments()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    if args.sleep < 0.0:
        raise ValueError("--sleep cannot be negative")

    if args.skip_stage75 and args.skip_stage76:
        raise ValueError(
            "Cannot skip both Stage 7.5 and Stage 7.6."
        )

    validate_executable(
        STAGE75_EXE,
        args.skip_stage75,
    )

    validate_executable(
        STAGE76_EXE,
        args.skip_stage76,
    )

    cases = QUICK_CASES if args.quick else DEFAULT_CASES
    repeats = 1 if args.quick else args.repeats

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    collect_environment()

    all_measurements: list[Measurement] = []

    for case in cases:
        for repeat in range(1, repeats + 1):
            programs = []

            if not args.skip_stage75:
                programs.append(
                    (STAGE75_EXE, "stage75")
                )

            if not args.skip_stage76:
                programs.append(
                    (STAGE76_EXE, "stage76")
                )

            # Alternate launch order to reduce thermal/order bias.
            if repeat % 2 == 0:
                programs.reverse()

            for executable, source in programs:
                measurements = run_one_program(
                    executable=executable,
                    source=source,
                    case=case,
                    repeat=repeat,
                )

                all_measurements.extend(measurements)

                if args.sleep > 0.0:
                    time.sleep(args.sleep)

    write_raw_csv(all_measurements)

    aggregate_rows = aggregate_measurements(
        all_measurements
    )

    write_summary_csv(aggregate_rows)

    write_summary_markdown(
        aggregate_rows,
        repeats=repeats,
    )

    print()
    print("Stage 7.7 benchmark completed.")
    print(f"Raw runs:     {OUTPUT_ROOT}")
    print(f"Raw summary:  results/stage77_benchmark_raw.csv")
    print(f"CSV summary:  {SUMMARY_CSV}")
    print(f"Markdown:     {SUMMARY_MD}")
    print(f"Environment:  {ENVIRONMENT_TXT}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
