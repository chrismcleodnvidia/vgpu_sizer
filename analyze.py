#!/usr/bin/env python3
"""
vGPU Sizing Analyzer — parses GPUProfiler CSV output and produces
a markdown sizing report for one of three scenarios:

  1. Baseline capture (physical workstation -> POC starting point)
  2. POC performance analysis
  3. Troubleshooting

Usage:
  python analyze.py --file data.csv --scenario 1
  python analyze.py --file data.csv --scenario 2 --current-profile 4Q
  python analyze.py --file data.csv --scenario 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sizing_data import (
    WORKLOADS,
    GPU_HARDWARE,
    FB_HEADROOM_FACTOR,
    classify_workload,
    get_ram,
    get_users_per_gpu,
    get_vcpu,
    recommend_profile,
)

# ─── Column matching ───────────────────────────────────────────────────
# GPUProfiler CSV format (actual columns):
#   Time (s), CPU (%), Mem (%), Mem Total (MB), Mem Used (MB),
#   Protocol (FPS), Protocol RTT (ms),
#   GPU0 (%), GPU0 Mem (%), GPU0 Encode (%), GPU0 Decode (%),
#   GPU0 Mem Total (MB), GPU0 Mem Used (MB)
#
# Matching is order-sensitive: more specific patterns are checked first
# to avoid "GPU0 (%)" matching before "GPU0 Mem (%)".

# Each entry: (metric_key, pattern, priority) — lower priority = matched first
COLUMN_RULES: list[tuple[str, re.Pattern, int]] = [
    # GPU-specific (must check before generic GPU/Mem patterns)
    ("gpu_fb_used",   re.compile(r"GPU\d+\s+Mem\s+Used", re.I),    10),
    ("gpu_fb_total",  re.compile(r"GPU\d+\s+Mem\s+Total", re.I),   10),
    # GPU0 Mem (%) is FB usage as percentage — NOT memory controller activity.
    # It's redundant with GPU0 Mem Used (MB), but we capture it for reference.
    ("gpu_fb_pct",    re.compile(r"GPU\d+\s+Mem\s+\(%\)", re.I),   10),
    ("nvenc",         re.compile(r"GPU\d+\s+Encode", re.I),         10),
    ("nvdec",         re.compile(r"GPU\d+\s+Decode", re.I),         10),
    ("gpu_util",      re.compile(r"GPU\d+\s+\(%\)", re.I),          10),
    # Protocol
    ("protocol_fps",  re.compile(r"Protocol.*FPS", re.I),           20),
    ("protocol_rtt",  re.compile(r"Protocol.*RTT|RTT|Latency", re.I), 20),
    # CPU
    ("cpu_util",      re.compile(r"^CPU\s+\(%\)", re.I),            20),
    # System memory (must NOT match GPU Mem)
    ("ram_used",      re.compile(r"^Mem\s+Used", re.I),             30),
    ("ram_total",     re.compile(r"^Mem\s+Total", re.I),            30),
    ("ram_pct",       re.compile(r"^Mem\s+\(%\)", re.I),            30),
    # Network
    ("network_tx",    re.compile(r"(?:Tx|Transmit|Network.*Tx)", re.I), 40),
    ("network_rx",    re.compile(r"(?:Rx|Receiv|Network.*Rx)", re.I),   40),
    # Display
    ("display_count", re.compile(r"Display.*Count", re.I),          40),
    ("resolution",    re.compile(r"Resolution", re.I),              40),
    # Fallback broader patterns for non-standard CSV exports
    ("gpu_util",      re.compile(r"gpu.*(?:util|usage)\s*\(%?\)", re.I), 50),
    ("gpu_fb_used",   re.compile(r"(?:fb|frame.?buffer|gpu.?mem).*(?:used|mb)", re.I), 50),
    ("cpu_util",      re.compile(r"cpu.*(?:util|usage|%)", re.I),   50),
    ("ram_used",      re.compile(r"(?:ram|sys.*mem).*(?:used|mb)", re.I), 50),
    ("nvenc",         re.compile(r"(?:nvenc|encoder)", re.I),       50),
    ("nvdec",         re.compile(r"(?:nvdec|decoder)", re.I),       50),
]


@dataclass
class SystemConfig:
    cpu_model: str = "Unknown"
    cpu_clock_ghz: float = 0.0
    gpu_model: str = "Unknown"
    total_ram_gb: float = 0.0
    driver_version: str = "Unknown"
    os_version: str = "Unknown"
    computer_name: str = "Unknown"
    display_count: int = 1
    max_resolution: str = "Unknown"
    vcpu_count: int = 0


@dataclass
class MetricStats:
    name: str
    unit: str
    min_val: float = 0.0
    avg_val: float = 0.0
    p50_val: float = 0.0
    p95_val: float = 0.0
    max_val: float = 0.0
    samples: int = 0


@dataclass
class AnalysisResult:
    config: SystemConfig
    metrics: dict[str, MetricStats] = field(default_factory=dict)
    workload_key: str = ""
    workload_reasoning: str = ""
    workload_confidence: float = 0.0
    recommended_profile: str = ""
    warnings: list[str] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    is_idle: bool = False
    gpu_data_present: bool = True


# ─── CSV Parsing ───────────────────────────────────────────────────────

def match_columns(headers: list[str]) -> dict[str, str]:
    """Match CSV column headers to metric keys. Returns {csv_header: metric_key}.
    Respects priority ordering so more specific patterns win."""
    col_map: dict[str, str] = {}  # csv_header -> metric_key
    assigned_keys: set[str] = set()  # metric_keys already assigned

    # Sort rules by priority (lower = first)
    sorted_rules = sorted(COLUMN_RULES, key=lambda r: r[2])

    for metric_key, pattern, _ in sorted_rules:
        if metric_key in assigned_keys:
            continue
        for h in headers:
            if h in col_map:
                continue
            if pattern.search(h):
                col_map[h] = metric_key
                assigned_keys.add(metric_key)
                break

    return col_map


def extract_system_config(rows: list[dict], headers: list[str],
                          column_data: dict[str, list[float]]) -> SystemConfig:
    """Extract system config from CSV metadata and static columns."""
    cfg = SystemConfig()
    # GPUProfiler CSV may have config in early rows or as comment lines.
    all_text = " ".join(" ".join(str(v) for v in row.values()) for row in rows[:20])
    all_text += " " + " ".join(headers)

    # CPU model + clock
    cpu_match = re.search(
        r"((?:Xeon|Core|Ryzen|EPYC|Threadripper)[^\n,;]{3,50})", all_text, re.I
    )
    if cpu_match:
        cfg.cpu_model = cpu_match.group(1).strip()
    clock_match = re.search(r"(\d+\.\d+)\s*GHz", all_text, re.I)
    if clock_match:
        cfg.cpu_clock_ghz = float(clock_match.group(1))

    # GPU model
    gpu_match = re.search(
        r"((?:NVIDIA|GeForce|Quadro|RTX|Tesla|A\d{2,3}|L\d{1,2}|P\d{1,4})[^\n,;]{2,60})",
        all_text, re.I,
    )
    if gpu_match:
        cfg.gpu_model = gpu_match.group(1).strip()

    # Driver — NVIDIA driver versions are like 535.129, 595.79 (3 digits . 2-3 digits)
    # Must avoid matching metric values like "100.00" or "3887.0"
    # Look for driver-like context or typical NVIDIA version ranges (3xx-6xx)
    drv_match = re.search(r"\b([3-6]\d{2}\.\d{2,3})\b", all_text)
    if drv_match:
        candidate = drv_match.group(1)
        # Exclude common metric values
        if float(candidate) > 200:
            cfg.driver_version = candidate

    # System RAM total from column data (constant value across rows)
    if "ram_total" in column_data and column_data["ram_total"]:
        cfg.total_ram_gb = column_data["ram_total"][0] / 1024  # MB -> GB

    # GPU FB total from column data
    if "gpu_fb_total" in column_data and column_data["gpu_fb_total"]:
        # Store as metadata (used for profile comparison)
        cfg._gpu_fb_total_mb = column_data["gpu_fb_total"][0]

    return cfg


def parse_csv(filepath: Path) -> tuple[dict[str, list[float]], list[str], list[dict]]:
    """
    Read a CSV file. Returns:
      - column_data: {metric_key: [float values]}
      - raw_headers: original header strings
      - raw_rows: list of row dicts
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        # Skip comment lines (some GPUProfiler CSVs have # prefixed metadata)
        lines = f.readlines()

    # Find the header row (first row with multiple comma-separated non-empty fields)
    header_idx = 0
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len([p for p in parts if p.strip()]) >= 3:
            header_idx = i
            break

    # Re-parse from header row
    reader = csv.DictReader(lines[header_idx:])
    raw_headers = reader.fieldnames or []

    # Map headers -> metric keys using priority-based matching
    col_map = match_columns(raw_headers)

    raw_rows: list[dict] = []
    column_data: dict[str, list[float]] = {mk: [] for mk in col_map.values()}

    for row in reader:
        raw_rows.append(row)
        for csv_h, mk in col_map.items():
            val_str = row.get(csv_h, "").strip()
            try:
                column_data[mk].append(float(val_str))
            except (ValueError, TypeError):
                pass

    return column_data, raw_headers, raw_rows


def compute_stats(values: list[float], name: str, unit: str = "%") -> MetricStats:
    if not values:
        return MetricStats(name=name, unit=unit)
    sv = sorted(values)
    n = len(sv)
    return MetricStats(
        name=name,
        unit=unit,
        min_val=sv[0],
        avg_val=sum(sv) / n,
        p50_val=sv[int(n * 0.5)],
        p95_val=sv[min(int(n * 0.95), n - 1)],
        max_val=sv[-1],
        samples=n,
    )




# ─── Anomaly Detection (Scenario 3) ───────────────────────────────────

def detect_anomalies(column_data: dict[str, list[float]], stats: dict[str, MetricStats]) -> list[dict]:
    """Find time-series anomalies: sustained highs, spikes, exhaustion events."""
    anomalies: list[dict] = []

    for key, values in column_data.items():
        if not values or key not in stats:
            continue
        st = stats[key]
        if st.samples < 5:
            continue

        threshold_high = st.avg_val + 2 * (st.p95_val - st.avg_val) if st.p95_val > st.avg_val else st.avg_val * 1.5

        # Detect sustained high periods (>= 5 consecutive samples above P95)
        run_start = None
        run_length = 0
        for i, v in enumerate(values):
            if v >= st.p95_val and st.p95_val > 0:
                if run_start is None:
                    run_start = i
                run_length += 1
            else:
                if run_length >= 5:
                    anomalies.append({
                        "metric": st.name,
                        "type": "sustained_high",
                        "start_sample": run_start,
                        "end_sample": run_start + run_length,
                        "value": f">{st.p95_val:.1f}{st.unit} for {run_length} samples",
                    })
                run_start = None
                run_length = 0
        if run_length >= 5:
            anomalies.append({
                "metric": st.name,
                "type": "sustained_high",
                "start_sample": run_start,
                "end_sample": run_start + run_length,
                "value": f">{st.p95_val:.1f}{st.unit} for {run_length} samples",
            })

        # Detect spikes (> threshold_high, isolated)
        for i, v in enumerate(values):
            if v > threshold_high and v > st.p95_val * 1.1:
                anomalies.append({
                    "metric": st.name,
                    "type": "spike",
                    "start_sample": i,
                    "end_sample": i,
                    "value": f"{v:.1f}{st.unit} (avg {st.avg_val:.1f})",
                })

    # Deduplicate overlapping anomalies
    seen = set()
    unique = []
    for a in anomalies:
        key = (a["metric"], a["type"], a["start_sample"])
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique[:30]  # Cap output


# ─── Idle / Insufficient Activity Detection ───────────────────────────

def detect_idle_capture(stats: dict[str, MetricStats]) -> tuple[bool, str]:
    """Detect if a capture appears idle or inactive.
    Returns (is_idle, explanation)."""
    gpu_st = stats.get("gpu_util")
    cpu_st = stats.get("cpu_util")
    fps_st = stats.get("protocol_fps")

    gpu_avg = gpu_st.avg_val if gpu_st and gpu_st.samples > 0 else 0
    cpu_avg = cpu_st.avg_val if cpu_st and cpu_st.samples > 0 else 0
    fps_avg = fps_st.avg_val if fps_st and fps_st.samples > 0 else 0

    if gpu_avg < 1 and cpu_avg < 10 and fps_avg < 2:
        return (True,
                f"Capture appears **idle or inactive** (CPU avg {cpu_avg:.1f}%, "
                f"GPU avg {gpu_avg:.1f}%, Protocol FPS avg {fps_avg:.1f}). "
                "Cannot produce reliable sizing from this data. "
                "Re-capture while the user is actively working in their application.")
    return (False, "")


def get_profile_series(profile: str) -> str:
    """Extract the series letter (A, B, Q, C) from a profile name like '4Q', '16A', '3B'."""
    if not profile:
        return ""
    m = re.search(r"(\d+)([ABQC])$", profile.upper())
    return m.group(2) if m else ""


def a_series_note(profile: str) -> str:
    """Return an A-series explanatory note if the profile is A-series, else empty string."""
    if get_profile_series(profile) != "A":
        return ""
    from sizing_data import PROFILE_SERIES_INFO
    info = PROFILE_SERIES_INFO["A"]
    return (
        f"\n> **A-series profile detected ({profile})**: {info['description']}\n"
        "> \n"
        "> Unlike B-series (virtual desktop) or Q-series (virtual workstation) profiles "
        "where each vGPU serves a single user, A-series vGPUs are designed for "
        "multi-session hosts. The frame buffer is shared across all user sessions on "
        "the RDSH host — **not dedicated per user**. Density planning should be based "
        "on concurrent sessions per host, not vGPU-to-user ratio.\n"
    )


# ─── Report Generation ─────────────────────────────────────────────────

def format_stats_table(stats: dict[str, MetricStats]) -> str:
    lines = ["| Metric | Min | Avg | P50 | P95 | Max | Samples |",
             "|--------|----:|----:|----:|----:|----:|--------:|"]
    for s in stats.values():
        # Skip metrics that are all zeros (no data captured)
        if s.samples > 0 and s.max_val > 0:
            lines.append(
                f"| {s.name} | {s.min_val:.1f}{s.unit} | {s.avg_val:.1f}{s.unit} | "
                f"{s.p50_val:.1f}{s.unit} | {s.p95_val:.1f}{s.unit} | {s.max_val:.1f}{s.unit} | {s.samples} |"
            )
    return "\n".join(lines)


def format_config_section(cfg: SystemConfig) -> str:
    lines = [
        "## System Configuration",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| CPU | {cfg.cpu_model} |",
        f"| CPU Clock | {cfg.cpu_clock_ghz:.2f} GHz |" if cfg.cpu_clock_ghz > 0 else f"| CPU Clock | Not detected |",
        f"| GPU | {cfg.gpu_model} |",
        f"| System RAM | {cfg.total_ram_gb:.0f} GB |" if cfg.total_ram_gb > 0 else f"| System RAM | Not detected |",
        f"| Driver | {cfg.driver_version} |",
        f"| Displays | {cfg.display_count} |",
        f"| Max Resolution | {cfg.max_resolution} |",
    ]
    return "\n".join(lines)


def clock_speed_assessment(cfg: SystemConfig, workload_key: str) -> str:
    wl = WORKLOADS[workload_key]
    lines = ["\n## CPU Clock Speed Assessment\n"]

    if cfg.cpu_clock_ghz <= 0:
        lines.append("**WARNING**: CPU clock speed not detected in profiler data. "
                      f"For {wl['label']} workloads, ensure the vGPU host CPU is >= {wl['min_clock_ghz']} GHz "
                      f"(preferred: {wl['preferred_clock_ghz']}+ GHz).\n")
        return "\n".join(lines)

    min_clk = wl["min_clock_ghz"]
    pref_clk = wl["preferred_clock_ghz"]

    if cfg.cpu_clock_ghz >= pref_clk:
        lines.append(f"Source CPU clock: **{cfg.cpu_clock_ghz:.2f} GHz** — meets the "
                      f"recommended minimum ({min_clk} GHz) for {wl['label']} workloads.")
    elif cfg.cpu_clock_ghz >= min_clk:
        lines.append(f"Source CPU clock: **{cfg.cpu_clock_ghz:.2f} GHz** — meets minimum "
                      f"({min_clk} GHz) but below preferred ({pref_clk}+ GHz).")
    else:
        lines.append(f"**WARNING**: Source CPU clock **{cfg.cpu_clock_ghz:.2f} GHz** is below "
                      f"the minimum recommended {min_clk} GHz for {wl['label']} workloads.")

    if wl["single_thread_sensitive"]:
        lines.append("")
        lines.append(f"**Single-threaded performance note**: {wl['label']} applications are "
                      "often single-thread-sensitive. Low aggregate CPU utilization does NOT mean "
                      "the CPU is adequate — a single thread may be clock-bound. "
                      f"Ensure the vGPU host CPU sustains >= {pref_clk} GHz per-core boost.")

    return "\n".join(lines)


def contention_assessment(result: "AnalysisResult", profile: str,
                          density: int | None) -> str:
    """Assess noisy-neighbour risk based on GPU util at various densities.

    NOTE: GPUProfiler captures do NOT include scheduler type — that information
    is only available via `nvidia-smi vgpu -ss` on the hypervisor host.
    This assessment presents the impact under each scheduler policy without
    assuming which one is active.
    """
    gpu_st = result.metrics.get("gpu_util")
    if not gpu_st or gpu_st.max_val == 0:
        return ""

    lines = ["\n## Contention / Noisy Neighbour Assessment\n"]

    # Determine max vGPUs per GPU for this profile across known hardware
    max_density = None
    for gpu_name, hw in GPU_HARDWARE.items():
        d = hw["profiles"].get(profile)
        if d and (max_density is None or d > max_density):
            max_density = d

    # Contextual note about scheduler uncertainty
    lines.append("> GPU utilization represents percentage of the **physical GPU's** "
                 "processing capacity used by this vGPU. vGPUs share the GPU engine "
                 "via time-sliced scheduling — each vGPU runs serially.\n")
    lines.append("> **Note**: Scheduler type (best-effort, equal share, or fixed share) "
                 "is not captured by GPUProfiler. To determine the active scheduler, "
                 "run `nvidia-smi vgpu -ss` on the hypervisor host. The default is "
                 "best-effort for all GPU architectures. Impact analysis below covers "
                 "all three policies.\n")

    # Check if this is a rendering/compute workload where high GPU util is expected
    is_sustained_gpu_workload = result.workload_key in ("rendering", "gpu_compute", "omniverse")

    if gpu_st.p95_val > 70:
        if is_sustained_gpu_workload:
            lines.append(f"GPU utilization P95 is **{gpu_st.p95_val:.0f}%** — this is "
                         f"**expected for {WORKLOADS[result.workload_key]['label']}** workloads. "
                         "High GPU utilization is normal and indicates the application is "
                         "actively using GPU resources as designed.\n")
            lines.append("**Density implication**: This workload will have **low vGPU density** "
                         "potential. Co-locating with other GPU-intensive sessions will cause "
                         "contention under any scheduler policy.\n")
        else:
            lines.append(f"**NOISY NEIGHBOUR RISK**: GPU utilization P95 is "
                         f"**{gpu_st.p95_val:.0f}%** of the physical GPU.\n")

        lines.append("**Impact by scheduler policy:**")
        lines.append(f"- **Best Effort** (default): This vGPU consumes "
                     f"{gpu_st.p95_val:.0f}% of GPU time, leaving only "
                     f"{100 - gpu_st.p95_val:.0f}% for all other vGPUs on the same "
                     "physical GPU.")
        if max_density and max_density > 1:
            fair_share = 100.0 / max_density
            lines.append(f"- **Equal Share**: Each of up to {max_density} vGPUs "
                         f"({profile}) gets ~{fair_share:.0f}% of GPU time. This vGPU "
                         f"needs {gpu_st.p95_val:.0f}% — it would be "
                         f"{'severely throttled' if gpu_st.p95_val > fair_share * 2 else 'throttled'}.")
            lines.append(f"- **Fixed Share**: Similar cap at ~{fair_share:.0f}% "
                         f"per {profile} vGPU.")

        lines.append("\n**Recommendations** (choose based on deployment needs):")
        lines.append("1. **Larger profile** — fewer vGPUs per physical GPU = more "
                     "GPU time per session")
        lines.append("2. **Dedicated GPU** (passthrough or 1:1 profile) — if workload "
                     "consistently needs >70% GPU")
        if not is_sustained_gpu_workload:
            lines.append("3. **Scheduler policy change** — switch from best-effort to "
                         "equal/fixed share to prevent starvation of co-located users")
        lines.append(f"{'3' if is_sustained_gpu_workload else '4'}. **Workload isolation** — "
                     "place rendering/compute vGPUs on separate physical GPUs from "
                     "interactive design sessions")

        if density and density > 1:
            lines.append(f"\n*At current density ({density}:1), this session at "
                         f"{gpu_st.p95_val:.0f}% GPU leaves {100 - gpu_st.p95_val:.0f}% "
                         f"for the remaining {density - 1} co-located sessions.*")

        # Allocation policy context
        lines.append("\n*GPU allocation policy also affects contention: "
                     "**breadth-first** (VMware default) spreads VMs for better "
                     "performance; **depth-first** (XenServer default) packs GPUs, "
                     "increasing contention risk.*")

    elif gpu_st.p95_val > 50:
        lines.append(f"GPU utilization P95 is **{gpu_st.p95_val:.0f}%**. Not yet at "
                     "noisy-neighbour risk for isolated testing, but monitor at "
                     "production density — under best-effort scheduling (the default), "
                     "this session consumes significant GPU time that co-located users "
                     "cannot use.")
    else:
        lines.append(f"GPU utilization P95 is **{gpu_st.p95_val:.0f}%** — low "
                     "contention risk at typical densities.")

    return "\n".join(lines)


def vcpu_ram_assessment(result: "AnalysisResult") -> str:
    """Assess vCPU and RAM adequacy against workload recommendations."""
    wl = WORKLOADS[result.workload_key]
    lines = ["\n## vCPU and RAM Assessment\n"]

    cpu_st = result.metrics.get("cpu_util")
    ram_st = result.metrics.get("ram_used")
    ram_pct_st = result.metrics.get("ram_pct")
    gpu_st = result.metrics.get("gpu_util")

    # vCPU assessment
    wl_vcpu = wl["vcpu"]
    wl_vcpu_min = wl_vcpu if isinstance(wl_vcpu, int) else wl_vcpu[0]
    wl_vcpu_max = wl_vcpu if isinstance(wl_vcpu, int) else wl_vcpu[1]

    if result.config.vcpu_count > 0:
        lines.append(f"**Allocated vCPUs**: {result.config.vcpu_count}")
        lines.append(f"**Workload recommendation**: {get_vcpu(result.workload_key)} "
                     f"vCPUs ({wl['label']})\n")

        if result.config.vcpu_count < wl_vcpu_min:
            lines.append(f"- **WARNING**: Allocated vCPUs "
                         f"({result.config.vcpu_count}) is below the minimum "
                         f"recommendation ({wl_vcpu_min}) for {wl['label']} workloads.")
        else:
            lines.append(f"- Allocated vCPUs ({result.config.vcpu_count}) meets the "
                         "recommendation.")

        if cpu_st and cpu_st.p95_val > 80:
            suggested = min(result.config.vcpu_count + 2, wl_vcpu_max + 4)
            lines.append(f"- CPU utilization P95 is {cpu_st.p95_val:.0f}% — consider "
                         f"increasing vCPU allocation to {suggested}.")
            # NUMA hint
            gpu_val = gpu_st.p95_val if gpu_st and gpu_st.samples > 0 else 0
            if gpu_val < 30:
                lines.append("  - *High CPU with low GPU utilization may also "
                             "indicate cross-NUMA scheduling. Verify vCPU pinning "
                             "to the same socket as the vGPU's physical GPU.*")
        elif cpu_st and cpu_st.p95_val < 30 and result.config.vcpu_count > wl_vcpu_min:
            lines.append(f"- CPU utilization P95 is {cpu_st.p95_val:.0f}% — current "
                         "allocation may be oversized (but verify single-thread "
                         "sensitivity for this workload type).")
    else:
        lines.append(f"**Allocated vCPUs**: Not provided "
                     "(use `--vcpu-count` to enable assessment)")
        lines.append(f"**Workload recommendation**: {get_vcpu(result.workload_key)} "
                     "vCPUs\n")
        if cpu_st:
            if cpu_st.p95_val > 80:
                lines.append(f"- CPU P95 at {cpu_st.p95_val:.0f}% — may need more "
                             "vCPUs.")
            else:
                lines.append(f"- CPU P95 at {cpu_st.p95_val:.0f}% — appears adequate.")

    # RAM assessment
    wl_ram = wl["ram_gb"]
    lines.append("")

    if result.config.total_ram_gb > 0:
        lines.append(f"**Allocated RAM**: {result.config.total_ram_gb:.0f} GB")
        lines.append(f"**Workload recommendation**: {get_ram(result.workload_key)}\n")

        if result.config.total_ram_gb < wl_ram[0]:
            lines.append(f"- **WARNING**: Allocated RAM "
                         f"({result.config.total_ram_gb:.0f} GB) is below the "
                         f"minimum recommendation ({wl_ram[0]} GB) for "
                         f"{wl['label']} workloads.")
        else:
            lines.append(f"- Allocated RAM ({result.config.total_ram_gb:.0f} GB) "
                         "meets the recommendation.")

        if ram_pct_st and ram_pct_st.p95_val > 85:
            lines.append(f"- RAM utilization P95 is {ram_pct_st.p95_val:.0f}% — "
                         "consider increasing RAM.")
        elif ram_st and ram_st.samples > 0 and result.config.total_ram_gb > 0:
            ram_used_gb = ram_st.p95_val / 1024  # MB -> GB
            ram_pct = (ram_used_gb / result.config.total_ram_gb) * 100
            if ram_pct > 85:
                lines.append(f"- RAM P95 usage: {ram_used_gb:.1f} GB / "
                             f"{result.config.total_ram_gb:.0f} GB ({ram_pct:.0f}%) "
                             "— consider increasing RAM.")
            else:
                lines.append(f"- RAM P95 usage: {ram_used_gb:.1f} GB / "
                             f"{result.config.total_ram_gb:.0f} GB ({ram_pct:.0f}%) "
                             "— adequate headroom.")
    else:
        lines.append("**Allocated RAM**: Not detected "
                     "(use `--total-ram` to enable assessment)")
        lines.append(f"**Workload recommendation**: {get_ram(result.workload_key)}")

    return "\n".join(lines)


def server_rollup_table(profile: str, workload_key: str, user_counts: list[int] | None = None) -> str:
    if user_counts is None:
        user_counts = [25, 50, 100]

    wl = WORKLOADS[workload_key]
    vcpu_val = wl["vcpu"]
    vcpu_per = vcpu_val if isinstance(vcpu_val, int) else vcpu_val[1]  # use upper end
    oversub = wl["cpu_oversub"]
    ram_per = wl["ram_gb"][1]  # upper end

    lines = [
        "\n## Server-Level Sizing Estimates\n",
        "*These are starting-point estimates. Always validate with a POC.*\n",
        "| Concurrent Users | GPUs Needed | GPU Model | Total Server RAM | Physical CPU Cores | Notes |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for gpu_model in WORKLOADS[workload_key]["recommended_gpus"]:
        upg = get_users_per_gpu(profile, gpu_model)
        if not upg:
            continue
        for n in user_counts:
            gpus = math.ceil(n / upg)
            total_ram = n * ram_per + 16  # OS overhead
            total_vcpus = n * vcpu_per
            phys_cores = math.ceil(total_vcpus / oversub)
            lines.append(
                f"| {n} | {gpus}x {gpu_model} | {gpu_model} | {total_ram} GB | "
                f"{phys_cores} | {oversub}:1 oversub, {profile} profile |"
            )

    return "\n".join(lines)


# ─── Scenario Reports ──────────────────────────────────────────────────

def report_scenario_1(result: AnalysisResult) -> str:
    """Baseline -> POC starting point."""
    from sizing_data import VGPU_PROFILES
    wl = WORKLOADS[result.workload_key]
    sections = [
        f"# vGPU Sizing Report — Baseline Analysis\n",
        f"*Scenario 1: Physical workstation capture -> POC starting point*\n",
        "---\n",
        format_config_section(result.config),
        "\n## Performance Metrics Summary\n",
        format_stats_table(result.metrics),
        f"\n## Workload Classification\n",
        f"**{wl['label']}** — {result.workload_reasoning}",
    ]
    if result.workload_confidence < 0.6:
        sections.append(f"\n*Confidence: {result.workload_confidence:.0%} — "
                        "consider specifying workload type explicitly with `--workload`.*")

    # Idle capture warning
    if result.is_idle:
        sections.append("\n## WARNING: Insufficient Activity\n")
        sections.append(result.warnings[0])
        sections.append("\n*Recommendations below are unreliable due to "
                        "insufficient activity.*\n")

    # No-GPU data warning
    if not result.gpu_data_present:
        sections.append("\n> **CPU-only baseline capture** (no GPU assigned). This is "
                        "valuable for comparing CPU-only vs GPU-accelerated performance. "
                        "CPU, RAM, and protocol metrics reflect the workload without "
                        "GPU offload. Pair with a GPU-enabled capture to quantify the "
                        "benefit of GPU acceleration and determine FB sizing.\n")

    sections.append(clock_speed_assessment(result.config, result.workload_key))

    # Profile recommendation
    sections.append(f"\n## Recommended vGPU Configuration\n")
    if result.is_idle:
        sections.append("**No recommendation** — insufficient activity data. "
                        "Re-capture with active user workload.")
    else:
        sections.append(f"| Setting | Recommendation |")
        sections.append(f"|---------|---------------|")
        sections.append(f"| **vGPU Profile** | {result.recommended_profile} |")
        series = wl['profile_series']
        license_label = wl['license']
        from sizing_data import PROFILE_SERIES_INFO
        series_desc = PROFILE_SERIES_INFO.get(series, {}).get("label", "")
        series_suffix = f" — {series_desc}" if series_desc else ""
        sections.append(f"| **License** | {license_label} ({series}-series{series_suffix}) |")
        sections.append(f"| **vCPU / session** | {get_vcpu(result.workload_key)} |")
        sections.append(f"| **CPU Oversub** | {wl['cpu_oversub']}:1 |")
        sections.append(f"| **Min Host CPU Clock** | {wl['min_clock_ghz']} GHz (prefer {wl['preferred_clock_ghz']}+) |")
        sections.append(f"| **RAM / session** | {get_ram(result.workload_key)} |")
        sections.append(f"| **Recommended GPU** | {', '.join(wl['recommended_gpus'])} |")

    # Display caveat
    if result.config.display_count > 1 or "4k" in result.config.max_resolution.lower() or "3840" in result.config.max_resolution:
        sections.append(f"\n**Display note**: {result.config.display_count} display(s) at {result.config.max_resolution}. "
                        "Multi-monitor or 4K may increase FB requirements — consider bumping to next profile size.")

    # Users per GPU table
    sections.append(f"\n## GPU Density Estimates\n")
    sections.append(f"| GPU Model | Profile | Max Users/GPU |")
    sections.append(f"|-----------|---------|:---:|")
    for gpu_model in wl["recommended_gpus"]:
        upg = get_users_per_gpu(result.recommended_profile, gpu_model)
        if upg:
            sections.append(f"| {gpu_model} | {result.recommended_profile} | {upg} |")

    # P95 GPU util density adjustment
    gpu_st = result.metrics.get("gpu_util")
    if gpu_st and gpu_st.p95_val > 60:
        if result.workload_key in ("rendering", "gpu_compute", "omniverse"):
            sections.append(f"\n**Density note**: P95 GPU utilization is {gpu_st.p95_val:.0f}% — "
                            f"this is **expected** for {wl['label']} workloads. "
                            "High GPU utilization means this workload will have **low vGPU density**. "
                            "Plan for 1:1 or low-ratio GPU allocation (dedicated GPU or large profile).")
        else:
            sections.append(f"\n**Density warning**: P95 GPU utilization is {gpu_st.p95_val:.0f}%. "
                            "Reduce density by ~25% from max to ensure headroom.")

    sections.append(server_rollup_table(result.recommended_profile, result.workload_key))

    # Encode/Decode assessment
    nvenc_st = result.metrics.get("nvenc")
    nvdec_st = result.metrics.get("nvdec")
    has_encdec = (nvenc_st and nvenc_st.max_val > 0) or (nvdec_st and nvdec_st.max_val > 0)
    if has_encdec:
        sections.append("\n## Encode / Decode Assessment\n")
        if nvenc_st and nvenc_st.max_val > 0:
            sections.append(f"- **NVENC (Encode)**: avg {nvenc_st.avg_val:.0f}%, "
                            f"P95 {nvenc_st.p95_val:.0f}%, max {nvenc_st.max_val:.0f}%")
            if nvenc_st.p95_val > 70:
                sections.append("  - **Warning**: Encoder near saturation — remoting protocol "
                                "frame delivery will degrade at this density. Consider H.265 "
                                "encoding or reducing max FPS.")
            elif nvenc_st.p95_val > 30:
                sections.append("  - Moderate encode load — factor this into density planning, "
                                "especially for multi-monitor or high-resolution sessions.")
        if nvdec_st and nvdec_st.max_val > 0:
            sections.append(f"- **NVDEC (Decode)**: avg {nvdec_st.avg_val:.0f}%, "
                            f"P95 {nvdec_st.p95_val:.0f}%, max {nvdec_st.max_val:.0f}%")
            if nvdec_st.p95_val > 70:
                sections.append("  - **Warning**: Decoder near saturation — video playback "
                                "or media-heavy workflows consuming significant decode capacity.")
            elif nvdec_st.p95_val > 30:
                sections.append("  - Moderate decode load — indicates video content or media "
                                "playback in the workflow.")

    # ── Omniverse-specific section ──
    if result.workload_key == "omniverse":
        sections.append("\n## Omniverse-Specific Requirements\n")
        sections.append("**RTX GPU is REQUIRED** -- RT Cores and Tensor Cores are mandatory for Omniverse.\n")

        # RAM rule: 2.5x VRAM
        profile_fb = VGPU_PROFILES.get(result.recommended_profile, 0)
        ram_min = profile_fb * 2.5
        sections.append(f"**System RAM**: Minimum **{ram_min:.0f} GB** (2.5x the {profile_fb} GB "
                        f"frame buffer). ECC memory recommended for critical workflows.\n")

        sections.append("**Storage**: 1 TB per user recommended.\n")
        sections.append("**Networking**: 10 Gb/s minimum per node for Design Collaboration workflows.\n")

        # Omniverse-specific vGPU profile table
        ov_profiles = wl.get("vgpu_profiles", {})
        if ov_profiles:
            sections.append("### Omniverse vGPU Profile Recommendations\n")
            sections.append("| GPU | Min Profile | Recommended Profile | User Density |")
            sections.append("|-----|------------|-------------------|:---:|")
            for gpu_name, info in ov_profiles.items():
                sections.append(f"| {gpu_name} | {info['min']} | {info['recommended']} | {info['density']} |")

        sections.append("\n### Sizing Considerations\n")
        sections.append("- **Render-heavy** workflows (USD Composer, Create): need more CPU cores + larger VRAM")
        sections.append("- **Simulation-heavy** workflows (Isaac Sim, Physics): need higher CPU clock + larger VRAM")
        sections.append("- Textures + geometry must fit entirely in VRAM or they **will not render**")
        sections.append("- Physics engine also consumes GPU resources -- ensure headroom beyond rendering")
        sections.append("- Multi-GPU rendering scaling is **nonlinear**; validate with POC")
        sections.append("- Third-party apps alongside Omniverse (Rhino, Maya, etc.) add to CPU/RAM/storage requirements")

    # RTX/Tensor flag (non-Omniverse workloads)
    elif result.workload_key != "omniverse":
        nvenc_st = result.metrics.get("nvenc")
        if nvenc_st and nvenc_st.avg_val > 20:
            sections.append(f"\n**RTX/Tensor core note**: Average NVENC utilization is {nvenc_st.avg_val:.0f}%. "
                            "Confirm whether this workload uses RTX or Tensor cores (e.g., AI inference, "
                            "ray tracing). This affects GPU model selection.")

    # Contention assessment
    sections.append(contention_assessment(result, result.recommended_profile, None))

    # Warnings
    if result.warnings:
        sections.append("\n## Warnings\n")
        for w in result.warnings:
            sections.append(f"- {w}")

    # Notes (skip for Omniverse since they're already shown in the dedicated section)
    if result.workload_key != "omniverse":
        sections.append(f"\n{wl['notes']}")
    sections.append("\n---\n*Report generated by vGPU Sizer. Always validate with a POC.*")
    return "\n".join(sections)


def report_scenario_2(result: AnalysisResult, current_profile: str, density: int | None) -> str:
    """POC performance analysis."""
    from sizing_data import VGPU_PROFILES
    wl = WORKLOADS[result.workload_key]

    # Determine profile FB size
    # For full GPU (physical/passthrough), extract from CSV total column
    is_full_gpu = "full gpu" in current_profile.lower() if current_profile else False
    if is_full_gpu or VGPU_PROFILES.get(current_profile, 0) == 0:
        fb_total_st = result.metrics.get("gpu_fb_total")
        if fb_total_st and fb_total_st.max_val > 0:
            profile_fb = fb_total_st.max_val / 1024  # MB -> GB
        elif hasattr(result.config, '_gpu_fb_total_mb'):
            profile_fb = result.config._gpu_fb_total_mb / 1024
        else:
            profile_fb = 0
        is_full_gpu = True
    else:
        profile_fb = VGPU_PROFILES.get(current_profile, 0)

    sections = [
        f"# vGPU Sizing Report — POC Performance Analysis\n",
        f"*Scenario 2: Assessing current vGPU configuration*\n",
        "---\n",
        format_config_section(result.config),
        f"\n**Current Profile**: {current_profile} ({profile_fb} GB FB)",
    ]
    # A-series context note
    a_note = a_series_note(current_profile)
    if a_note:
        sections.append(a_note)
    if density:
        sections.append(f"**Current Density**: {density} users/GPU")

    sections.append("\n## Performance Metrics Summary\n")
    sections.append(format_stats_table(result.metrics))

    # Adequacy assessment
    sections.append("\n## Profile Adequacy Assessment\n")

    fb_st = result.metrics.get("gpu_fb_used")
    gpu_st = result.metrics.get("gpu_util")
    nvenc_st = result.metrics.get("nvenc")
    cpu_st = result.metrics.get("cpu_util")

    bottlenecks: list[str] = []

    if fb_st and profile_fb > 0:
        # Convert profile FB to same unit as metric (MB)
        profile_fb_mb = profile_fb * 1024 if "MB" in fb_st.unit.upper() or fb_st.max_val > 100 else profile_fb
        fb_headroom = (profile_fb_mb - fb_st.max_val) / profile_fb_mb * 100
        sections.append(f"- **Frame Buffer headroom**: {fb_headroom:.0f}% "
                        f"(peak {fb_st.max_val:.0f}{fb_st.unit} / {profile_fb} GB profile)")
        if fb_headroom < 15:
            bottlenecks.append(f"Frame buffer near capacity ({fb_headroom:.0f}% headroom) — consider upsizing profile")
        elif fb_headroom > 60:
            bottlenecks.append(f"Frame buffer significantly underutilized ({fb_headroom:.0f}% headroom) — consider downsizing profile")

    if gpu_st:
        sections.append(f"- **GPU utilization headroom**: {100 - gpu_st.p95_val:.0f}% "
                        f"(P95: {gpu_st.p95_val:.0f}%)")
        is_sustained = result.workload_key in ("rendering", "gpu_compute", "omniverse")
        if gpu_st.p95_val > 80:
            if is_sustained:
                bottlenecks.append(f"GPU utilization P95 at {gpu_st.p95_val:.0f}% — "
                                   f"expected for {WORKLOADS[result.workload_key]['label']} workloads. "
                                   "Low vGPU density — plan for dedicated or 1:1 GPU allocation")
            else:
                bottlenecks.append(f"GPU utilization P95 at {gpu_st.p95_val:.0f}% — GPU engine saturated")
        elif gpu_st.p95_val > 60:
            if is_sustained:
                bottlenecks.append(f"GPU utilization P95 at {gpu_st.p95_val:.0f}% — "
                                   f"moderate for {WORKLOADS[result.workload_key]['label']}. "
                                   "Monitor at production density")
            else:
                bottlenecks.append(f"GPU utilization P95 at {gpu_st.p95_val:.0f}% — approaching saturation")

    nvdec_st = result.metrics.get("nvdec")

    if nvenc_st and nvenc_st.max_val > 0:
        sections.append(f"- **NVENC (Encode) utilization**: avg {nvenc_st.avg_val:.0f}%, "
                        f"P95 {nvenc_st.p95_val:.0f}%")
        if nvenc_st.p95_val > 70:
            bottlenecks.append(f"NVENC P95 at {nvenc_st.p95_val:.0f}% — encoder saturated, "
                               "may cause frame drops or protocol lag")
        elif nvenc_st.p95_val > 50:
            bottlenecks.append(f"NVENC P95 at {nvenc_st.p95_val:.0f}% — encoder under moderate load, "
                               "monitor at higher density")

    if nvdec_st and nvdec_st.max_val > 0:
        sections.append(f"- **NVDEC (Decode) utilization**: avg {nvdec_st.avg_val:.0f}%, "
                        f"P95 {nvdec_st.p95_val:.0f}%")
        if nvdec_st.p95_val > 70:
            bottlenecks.append(f"NVDEC P95 at {nvdec_st.p95_val:.0f}% — decoder saturated, "
                               "video playback or media-heavy workflow consuming decode capacity")
        elif nvdec_st.p95_val > 50:
            bottlenecks.append(f"NVDEC P95 at {nvdec_st.p95_val:.0f}% — decoder under moderate load")

    if cpu_st and cpu_st.p95_val > 85:
        bottlenecks.append(f"CPU P95 at {cpu_st.p95_val:.0f}% — CPU starvation risk")

    sections.append(clock_speed_assessment(result.config, result.workload_key))

    # Idle capture warning
    if result.is_idle:
        sections.append("\n## WARNING: Insufficient Activity\n")
        sections.append(result.warnings[0])
        sections.append("\n*Profile recommendations below are unreliable due to "
                        "insufficient activity.*\n")

    # No-GPU data note
    if not result.gpu_data_present:
        sections.append("\n> **CPU-only baseline capture** (no GPU assigned). This is "
                        "valuable for comparing CPU-only vs GPU-accelerated performance. "
                        "CPU, RAM, and protocol metrics reflect the workload without "
                        "GPU offload. Pair with a GPU-enabled capture to quantify the "
                        "benefit of GPU acceleration and determine FB sizing.\n")

    # Bottleneck summary
    sections.append("\n## Bottleneck Analysis\n")
    if bottlenecks:
        for b in bottlenecks:
            sections.append(f"- **{b}**")
    else:
        sections.append("No significant bottlenecks detected. Current configuration "
                        "appears adequate.")

    # Contention assessment
    sections.append(contention_assessment(result, current_profile, density))

    # Recommendation — metric-driven for scenario 2
    sections.append("\n## Recommendation\n")

    if result.is_idle:
        sections.append("**No recommendation** — insufficient activity data. "
                        "Re-capture with active user workload.")
    else:
        # Assess adequacy from actual metrics, not just FB math
        fb_headroom_pct = None
        fb_adequate = True
        gpu_adequate = True

        if fb_st and profile_fb > 0:
            profile_fb_mb = profile_fb * 1024 if fb_st.max_val > 100 else profile_fb
            fb_headroom_pct = (profile_fb_mb - fb_st.max_val) / profile_fb_mb * 100
            if fb_headroom_pct < 15:
                fb_adequate = False

        if gpu_st and gpu_st.p95_val > 80:
            gpu_adequate = False

        if fb_adequate and gpu_adequate:
            sections.append(f"Current profile **{current_profile}** ({profile_fb} GB) "
                            "is adequate for this workload.")
            if fb_headroom_pct is not None:
                sections.append(f"  - FB headroom: {fb_headroom_pct:.0f}% — sufficient "
                                "for interactive spikes.")
            if gpu_st:
                sections.append(f"  - GPU util P95: {gpu_st.p95_val:.0f}% — within "
                                "acceptable range.")
            # Check if downsizing is possible
            optimal = result.recommended_profile
            if VGPU_PROFILES.get(optimal, 0) < profile_fb:
                sections.append(f"\n  *Density optimization*: Metrics suggest "
                                f"**{optimal}** ({VGPU_PROFILES.get(optimal, '?')} GB) "
                                "may also work, increasing density. Validate with "
                                "representative peak workloads before downsizing.")
        elif not fb_adequate:
            optimal = result.recommended_profile
            sections.append(f"**Upsize** from {current_profile} to **{optimal}** "
                            f"({VGPU_PROFILES.get(optimal, '?')} GB) — frame buffer "
                            f"headroom is insufficient "
                            f"({fb_headroom_pct:.0f}%).")
        else:
            # GPU saturated but FB is OK
            if result.workload_key in ("rendering", "gpu_compute", "omniverse"):
                sections.append(f"GPU engine P95 at **{gpu_st.p95_val:.0f}%** — this is "
                                f"expected for {WORKLOADS[result.workload_key]['label']} "
                                "workloads. Current profile FB is adequate, but high GPU "
                                "utilization means **low density potential**. Consider a "
                                "dedicated GPU (1:1 profile or passthrough) for this workload.")
            else:
                sections.append(f"**Consider upsizing** from {current_profile} — GPU "
                                f"engine P95 at {gpu_st.p95_val:.0f}% indicates "
                                "saturation. A larger profile reduces density and gives "
                                "this vGPU more GPU time per scheduling cycle.")

    # vCPU and RAM assessment
    sections.append(vcpu_ram_assessment(result))

    # Protocol metrics (only show if non-zero data present)
    fps_st = result.metrics.get("protocol_fps")
    rtt_st = result.metrics.get("protocol_rtt")
    has_protocol = (fps_st and fps_st.max_val > 0) or (rtt_st and rtt_st.max_val > 0)
    if has_protocol:
        sections.append("\n## Protocol Performance\n")
        if fps_st and fps_st.max_val > 0:
            sections.append(f"- FPS: avg {fps_st.avg_val:.0f}, P5 {fps_st.min_val:.0f}")
            # FRL detection
            for frl_cap in (30, 45, 60):
                if abs(fps_st.p50_val - frl_cap) <= 1 and fps_st.p50_val > 0:
                    series_note = ("B-series default: 45 FPS, Q-series/A-series default: 60 FPS"
                                   if frl_cap in (45, 60)
                                   else "may be a custom FRL setting")
                    sections.append(f"  - *FPS appears capped at ~{frl_cap} — likely "
                                    f"frame rate limiter ({series_note}). FRL is "
                                    "disabled when using equal/fixed share scheduler.*")
                    break
        if rtt_st and rtt_st.max_val > 0:
            sections.append(f"- RTT/Latency: avg {rtt_st.avg_val:.0f}ms, "
                            f"P95 {rtt_st.p95_val:.0f}ms")

    sections.append("\n## Next Steps\n")
    sections.append("- [ ] Validate with representative user workflows")
    sections.append("- [ ] Test at target density with concurrent users")
    sections.append("- [ ] Monitor protocol metrics (FPS, RTT) during user "
                    "acceptance testing")
    if bottlenecks:
        sections.append("- [ ] Address identified bottlenecks before scaling")

    sections.append("\n---\n*Report generated by vGPU Sizer. Always validate with "
                    "a POC.*")
    return "\n".join(sections)


def report_scenario_3(result: AnalysisResult, symptom: str = "") -> str:
    """Troubleshooting."""
    wl = WORKLOADS[result.workload_key]

    sections = [
        f"# vGPU Sizing Report — Troubleshooting Analysis\n",
        f"*Scenario 3: Issue diagnosis*\n",
    ]
    if symptom:
        sections.append(f"**Reported symptom**: {symptom}\n")
    sections.append("---\n")
    sections.append(format_config_section(result.config))
    sections.append("\n## Performance Metrics Summary\n")
    sections.append(format_stats_table(result.metrics))

    # Anomalies
    sections.append("\n## Detected Anomalies\n")
    if result.anomalies:
        sections.append("| Metric | Type | Samples | Value |")
        sections.append("|--------|------|:---:|-------|")
        for a in result.anomalies[:20]:
            span = f"{a['start_sample']}-{a['end_sample']}" if a['start_sample'] != a['end_sample'] else str(a['start_sample'])
            sections.append(f"| {a['metric']} | {a['type']} | {span} | {a['value']} |")
    else:
        sections.append("No significant anomalies detected in the time-series data.")

    # Root cause analysis
    sections.append("\n## Probable Root Causes (Ranked)\n")
    causes: list[tuple[int, str, str]] = []  # (priority, cause, evidence)

    gpu_st = result.metrics.get("gpu_util")
    fb_st = result.metrics.get("gpu_fb_used")
    nvenc_st = result.metrics.get("nvenc")
    cpu_st = result.metrics.get("cpu_util")
    rtt_st = result.metrics.get("protocol_rtt")
    fps_st = result.metrics.get("protocol_fps")

    # FB exhaustion — compare against actual GPU FB total, not a hardcoded threshold
    fb_total_st = result.metrics.get("gpu_fb_total")
    gpu_fb_total_mb = fb_total_st.max_val if fb_total_st and fb_total_st.max_val > 0 else 0
    # Also check SystemConfig for GPU FB (from GPD extraction)
    if gpu_fb_total_mb == 0 and hasattr(result.config, '_gpu_fb_total_mb'):
        gpu_fb_total_mb = result.config._gpu_fb_total_mb

    if fb_st and fb_st.max_val > 0 and gpu_fb_total_mb > 0:
        fb_used_pct = (fb_st.p95_val / gpu_fb_total_mb) * 100
        if fb_used_pct > 90:
            causes.append((1, "Frame buffer exhaustion",
                           f"P95 FB usage: {fb_st.p95_val:.0f} MB / {gpu_fb_total_mb:.0f} MB "
                           f"({fb_used_pct:.0f}%). Frame buffer is critically full."))
        elif fb_used_pct > 75:
            causes.append((3, "Frame buffer pressure",
                           f"P95 FB usage: {fb_st.p95_val:.0f} MB / {gpu_fb_total_mb:.0f} MB "
                           f"({fb_used_pct:.0f}%). Usage is high but not exhausted. "
                           f"Headroom: {gpu_fb_total_mb - fb_st.p95_val:.0f} MB."))
    elif fb_st and fb_st.max_val > 0 and gpu_fb_total_mb == 0:
        # No total FB info available — note it without flagging exhaustion
        causes.append((4, "Frame buffer usage (total unknown)",
                       f"P95 FB usage: {fb_st.p95_val:.0f} MB, Max: {fb_st.max_val:.0f} MB. "
                       "GPU total FB not available — cannot determine headroom."))

    # GPU saturation
    if gpu_st and gpu_st.p95_val > 80:
        if result.workload_key in ("rendering", "gpu_compute", "omniverse"):
            causes.append((4, "GPU engine at capacity (expected for this workload)",
                           f"P95 GPU util: {gpu_st.p95_val:.0f}%. This is expected for "
                           f"{WORKLOADS[result.workload_key]['label']} workloads — high GPU "
                           "utilization is normal. If the user reports poor performance, the "
                           "root cause is likely elsewhere (CPU, RAM, or network)."))
        else:
            causes.append((2, "GPU engine saturation",
                           f"P95 GPU util: {gpu_st.p95_val:.0f}%. Engine is near capacity."))

    nvdec_st = result.metrics.get("nvdec")

    # NVENC bottleneck
    if nvenc_st and nvenc_st.p95_val > 60:
        causes.append((3, "NVENC encoder bottleneck",
                       f"P95 NVENC: {nvenc_st.p95_val:.0f}%. Heavy video encoding load — "
                       "remoting protocol frame delivery may be affected."))

    # NVDEC bottleneck
    if nvdec_st and nvdec_st.p95_val > 60:
        causes.append((3, "NVDEC decoder bottleneck",
                       f"P95 NVDEC: {nvdec_st.p95_val:.0f}%. Heavy video decode load — "
                       "media playback, video conferencing, or video editing consuming decode capacity."))

    # Single-threaded CPU (the subtle one)
    if cpu_st and cpu_st.avg_val < 30 and wl["single_thread_sensitive"]:
        clock_note = ""
        if result.config.cpu_clock_ghz > 0 and result.config.cpu_clock_ghz < wl["min_clock_ghz"]:
            clock_note = f" Host CPU clock ({result.config.cpu_clock_ghz:.2f} GHz) is below minimum ({wl['min_clock_ghz']} GHz)."
            causes.append((2, "CPU clock speed / single-threaded bottleneck",
                           f"CPU util is low ({cpu_st.avg_val:.0f}%) but {wl['label']} workloads are single-thread-sensitive."
                           f"{clock_note} A single core may be pegged at 100% while the aggregate looks fine."))
        else:
            causes.append((4, "Possible CPU clock speed / single-threaded bottleneck",
                           f"CPU util is low ({cpu_st.avg_val:.0f}%) — if user reports sluggishness, "
                           f"check host CPU per-core boost clock for {wl['label']} workloads (need >= {wl['min_clock_ghz']} GHz)."))

    # CPU starvation (high overall)
    if cpu_st and cpu_st.p95_val > 85:
        causes.append((2, "CPU starvation",
                       f"P95 CPU util: {cpu_st.p95_val:.0f}%. Not enough vCPU allocated or oversub too high."))

    # Network / protocol issues
    if rtt_st and rtt_st.p95_val > 100:
        causes.append((3, "Network latency",
                       f"P95 RTT: {rtt_st.p95_val:.0f}ms. High latency degrades user experience."))
    if fps_st and fps_st.avg_val < 15:
        causes.append((3, "Low protocol FPS",
                       f"Average FPS: {fps_st.avg_val:.0f}. Below 15 FPS indicates rendering or encoding bottleneck."))

    if not causes:
        causes.append((5, "No clear bottleneck in profiler data",
                       "Metrics appear within normal ranges. Issue may be application-specific, "
                       "network-related (check switch/firewall), or driver/config related. "
                       "Check driver version compatibility and vGPU license state."))

    causes.sort(key=lambda c: c[0])
    for _, cause, evidence in causes:
        sections.append(f"### {cause}\n")
        sections.append(f"{evidence}\n")

    sections.append(clock_speed_assessment(result.config, result.workload_key))

    # Remediation
    sections.append("\n## Recommended Remediation\n")
    for _, cause, _ in causes[:3]:
        if "frame buffer" in cause.lower():
            sections.append("- **Upsize vGPU profile** to next available FB size")
        elif "gpu engine" in cause.lower() or "gpu saturation" in cause.lower():
            sections.append("- **Reduce density** (fewer users per GPU) or upsize profile")
        elif "nvenc" in cause.lower():
            sections.append("- **Check remoting protocol encode settings** (H.264 vs H.265, quality slider)")
            sections.append("- Consider reducing max FPS in protocol settings")
        elif "nvdec" in cause.lower():
            sections.append("- **Review media/video playback** in the user session — video content consumes decode capacity")
            sections.append("- Consider offloading video decode to CPU if supported by protocol/application")
            sections.append("- Reduce density if decode contention across sessions")
        elif "clock speed" in cause.lower() or "single-threaded" in cause.lower():
            sections.append(f"- **Verify host CPU per-core boost clock** >= {wl['preferred_clock_ghz']} GHz")
            sections.append("- Consider hosts with higher clock speed CPUs (e.g., Intel Xeon Gold 63xx+ series)")
        elif "cpu starvation" in cause.lower():
            sections.append(f"- **Increase vCPU** per session (currently recommended: {get_vcpu(result.workload_key)})")
            sections.append(f"- **Reduce oversubscription** ratio (currently {wl['cpu_oversub']}:1 max)")
        elif "network" in cause.lower() or "latency" in cause.lower():
            sections.append("- **Check network path** (switch hops, firewall, QoS)")
            sections.append("- Verify < 10ms RTT for LAN, < 50ms for WAN")
        elif "fps" in cause.lower():
            sections.append("- **Check remoting protocol settings** and GPU encode capacity")

    sections.append("\n---\n*Report generated by vGPU Sizer. Always validate findings with additional testing.*")
    return "\n".join(sections)


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="vGPU Sizing Analyzer")
    parser.add_argument("--file", required=True, help="Path to GPUProfiler CSV file")
    parser.add_argument("--scenario", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--workload", choices=list(WORKLOADS.keys()), default=None)
    parser.add_argument("--current-profile", default=None, help="Current vGPU profile (e.g. 4Q)")
    parser.add_argument("--density", type=int, default=None, help="Current users per GPU")
    parser.add_argument("--gpu-model", default=None, help="Target GPU model")
    parser.add_argument("--output", default=None, help="Output file path (default: <input>_report.md)")
    parser.add_argument("--symptom", default="", help="Symptom description for scenario 3")
    # System config overrides (typically extracted from GPUProfiler screenshot by Claude)
    parser.add_argument("--cpu-model", default=None, help="CPU model string")
    parser.add_argument("--cpu-clock", type=float, default=None, help="CPU clock speed in GHz")
    parser.add_argument("--gpu-name", default=None, help="GPU model name")
    parser.add_argument("--vgpu-profile", default=None, help="Detected vGPU profile name (e.g. 4Q)")
    parser.add_argument("--driver", default=None, help="NVIDIA driver version")
    parser.add_argument("--total-ram", type=float, default=None, help="Total system RAM in GB")
    parser.add_argument("--display-info", default=None, help="Display info string (e.g. '2x 1920x1080')")
    parser.add_argument("--vcpu-count", type=int, default=None, help="Allocated vCPUs for this VM")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    # Handle GPD files — binary format, CSV export required
    if filepath.suffix.lower() == ".gpd":
        print("ERROR: This tool requires CSV input. GPD is a binary format.", file=sys.stderr)
        print("", file=sys.stderr)
        print("To get both system config and metrics from a GPD file:", file=sys.stderr)
        print("  1. Run: python screenshot_config.py --gpd <file.gpd> --output screenshot.png", file=sys.stderr)
        print("     This opens GPUProfiler and captures system config via screenshot.", file=sys.stderr)
        print("  2. Export CSV from the open GPUProfiler window (click Export/CSV button)", file=sys.stderr)
        print("  3. Re-run with: python analyze.py --file <exported.csv> [--config overrides]", file=sys.stderr)
        sys.exit(1)

    # Parse CSV
    column_data, raw_headers, raw_rows = parse_csv(filepath)

    if not column_data or all(len(v) == 0 for v in column_data.values()):
        print("ERROR: No recognized metric columns found in CSV.", file=sys.stderr)
        print(f"Headers found: {raw_headers[:20]}", file=sys.stderr)
        print("If column names don't match expected patterns, please open an issue.", file=sys.stderr)
        sys.exit(1)

    # Print matched columns for transparency
    print("Matched columns:", file=sys.stderr)
    for mk, vals in column_data.items():
        if vals:
            print(f"  {mk}: {len(vals)} samples", file=sys.stderr)

    # System config (auto-detected from CSV, then overridden by CLI args)
    config = extract_system_config(raw_rows, raw_headers, column_data)

    # Apply config overrides from CLI (typically from GPUProfiler screenshot)
    if args.cpu_model:
        config.cpu_model = args.cpu_model
    if args.cpu_clock is not None:
        config.cpu_clock_ghz = args.cpu_clock
    if args.gpu_name:
        config.gpu_model = args.gpu_name
    if args.driver:
        config.driver_version = args.driver
    if args.total_ram is not None:
        config.total_ram_gb = args.total_ram
    if args.display_info:
        config.max_resolution = args.display_info
        # Try to extract display count from string like "2x 1920x1080"
        import re as _re
        dc = _re.match(r"(\d+)\s*x\s", args.display_info)
        if dc:
            config.display_count = int(dc.group(1))

    if args.vcpu_count is not None:
        config.vcpu_count = args.vcpu_count

    # If --vgpu-profile provided, use it as --current-profile for scenario 2
    if args.vgpu_profile and not args.current_profile:
        args.current_profile = args.vgpu_profile

    # For scenario 2 without a profile: if we have GPU total FB from the CSV,
    # this is a full GPU (physical or passthrough). Use the total as the "profile".
    if args.scenario == 2 and not args.current_profile:
        fb_total = column_data.get("gpu_fb_total", [])
        if fb_total and fb_total[0] > 0:
            full_fb_gb = fb_total[0] / 1024
            args.current_profile = f"full GPU ({full_fb_gb:.0f} GB)"

    # Compute stats
    metric_names = {
        "gpu_util": ("GPU Utilization", "%"),
        "gpu_fb_used": ("GPU Frame Buffer Used", " MB"),
        # gpu_fb_pct is redundant with gpu_fb_used (MB) — skip from stats display
        "nvenc": ("NVENC Utilization", "%"),
        "nvdec": ("NVDEC Utilization", "%"),
        "cpu_util": ("CPU Utilization", "%"),
        "ram_used": ("System RAM Used", " MB"),
        "ram_pct": ("System RAM", "%"),
        "protocol_fps": ("Protocol FPS", " fps"),
        "protocol_rtt": ("Protocol RTT", " ms"),
        "network_tx": ("Network Tx", " KB/s"),
        "network_rx": ("Network Rx", " KB/s"),
    }

    stats: dict[str, MetricStats] = {}
    for key, values in column_data.items():
        if key in metric_names and values:
            name, unit = metric_names[key]
            stats[key] = compute_stats(values, name, unit)

    # Detect GPU data presence
    gpu_util_vals = column_data.get("gpu_util", [])
    gpu_fb_vals = column_data.get("gpu_fb_used", [])
    gpu_data_present = bool(
        (gpu_util_vals and max(gpu_util_vals, default=0) > 0) or
        (gpu_fb_vals and max(gpu_fb_vals, default=0) > 0)
    )

    # Workload classification
    avg_gpu = stats["gpu_util"].avg_val if "gpu_util" in stats else 0
    peak_fb_gb = (stats["gpu_fb_used"].max_val / 1024 if "gpu_fb_used" in stats
                  else 0)  # Assume MB input
    avg_nvenc = stats["nvenc"].avg_val if "nvenc" in stats else 0
    avg_cpu = stats["cpu_util"].avg_val if "cpu_util" in stats else 0

    if args.workload:
        workload_key = args.workload
        reasoning = "Specified by user"
        confidence = 1.0
    else:
        workload_key, reasoning, confidence = classify_workload(
            avg_gpu, peak_fb_gb, avg_nvenc, avg_cpu,
            filename=Path(args.file).stem,
            gpu_data_present=gpu_data_present,
        )

    # Idle detection
    is_idle, idle_msg = detect_idle_capture(stats)

    # Profile recommendation
    series = WORKLOADS[workload_key]["profile_series"]
    profile = recommend_profile(peak_fb_gb, series, workload_key)

    # Build result
    result = AnalysisResult(
        config=config,
        metrics=stats,
        workload_key=workload_key,
        workload_reasoning=reasoning,
        workload_confidence=confidence,
        recommended_profile=profile,
        is_idle=is_idle,
        gpu_data_present=gpu_data_present,
    )
    if is_idle:
        result.warnings.insert(0, idle_msg)

    # Scenario 3: detect anomalies
    if args.scenario == 3:
        result.anomalies = detect_anomalies(column_data, stats)

    # Generate report
    if args.scenario == 1:
        report = report_scenario_1(result)
    elif args.scenario == 2:
        if not args.current_profile:
            print("ERROR: --current-profile is required for scenario 2", file=sys.stderr)
            sys.exit(1)
        report = report_scenario_2(result, args.current_profile, args.density)
    else:
        report = report_scenario_3(result, args.symptom)

    # Output
    print(report)

    # Save to file
    output_path = args.output or str(filepath.with_name(filepath.stem + "_report.md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
