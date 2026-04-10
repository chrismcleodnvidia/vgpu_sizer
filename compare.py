#!/usr/bin/env python3
"""
Compare GPU vs no-GPU captures from the same environment.

Reads CSV files from a directory, identifies pairs/groups where the same
machine ran the same workload with and without a GPU (or at different
profile sizes), and produces a markdown comparison report.

Pairing logic:
  - Match by hostname (from config JSON or filename prefix like IBI15)
  - Match by workload keywords in filename (e.g., "sketchup", "revit", "multi app")
  - Group: no-GPU baseline vs GPU-enabled captures at various profile sizes

Usage:
  python compare.py --dir <csv_directory> [--configs <configs.json>] [--output <report.md>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Reuse analyze.py's CSV parsing and stats
sys.path.insert(0, str(Path(__file__).parent))
from analyze import parse_csv, compute_stats, MetricStats


@dataclass
class CaptureInfo:
    name: str
    csv_path: Path
    hostname: str = ""
    gpu_model: str = ""
    vgpu_profile: str = ""
    gpu_type: str = ""  # "vgpu", "physical_or_passthrough", or ""
    has_gpu_data: bool = False
    workload_keywords: list[str] = None
    metrics: dict[str, MetricStats] = None

    def __post_init__(self):
        if self.workload_keywords is None:
            self.workload_keywords = []
        if self.metrics is None:
            self.metrics = {}


# ─── Workload keyword extraction ──────────────────────────────────────

WORKLOAD_PATTERNS = [
    (r"sketchup", "sketchup"),
    (r"revit", "revit"),
    (r"enscape", "enscape"),
    (r"vray|v-ray|v_ray", "vray"),
    (r"3ds?\s*max|3dsmax", "3dsmax"),
    (r"multi\s*app", "multiapp"),
    (r"carla", "carla"),
    (r"beach\s*bar", "beachbar"),
    (r"bicester", "bicester"),
    (r"bacn", "bacn"),
    (r"clone", "clone"),
    (r"render", "render"),
]

# Hostname prefix pattern: IBI01, IBI14, etc. → normalise to machine group
HOSTNAME_PREFIX_RE = re.compile(r"^(IBI\d{1,2})", re.I)


def extract_workload_keywords(filename: str) -> list[str]:
    fn = filename.lower()
    kws = []
    for pattern, label in WORKLOAD_PATTERNS:
        if re.search(pattern, fn, re.I):
            kws.append(label)
    return kws


def extract_machine_group(filename: str, hostname: str) -> str:
    """Get a normalised machine group from hostname or filename prefix.

    Hostname mapping (from the Lee dataset):
      IBI01      -> IBI01       (direct)
      IBI4/IBI5/IBI6 -> IBI14/IBI15/IBI16 (short hostnames for VMs on those machines)
      IBIIC-1..5 -> IBIIC-N    (clone sessions — IC = Instant Clone)
      vdiclient01 -> vdiclient  (generic VDI client name)
    """
    # Build hostname -> machine mapping
    if hostname:
        hn = hostname.upper().strip()

        # IBIIC-N: instant clone session — map to parent machine
        ic_match = re.match(r"IBIIC-(\d+)", hn)
        if ic_match:
            # IBIIC-4 corresponds to IBI14, IBIIC-5 to IBI15, etc. based on the data
            n = int(ic_match.group(1))
            return f"IBI1{n}"

        # Short hostname like IBI4, IBI5, IBI6 -> IBI14, IBI15, IBI16
        short_match = re.match(r"^IBI(\d)$", hn)
        if short_match:
            return f"IBI1{short_match.group(1)}"

        # Full hostname like IBI01
        full_match = re.match(r"^(IBI\d{2})", hn)
        if full_match:
            return full_match.group(1)

        # vdiclient01 — generic, fall through to filename
        if "VDICLIENT" in hn:
            pass  # use filename instead
        else:
            return hn

    # From filename: IBI01, IBI14, IBI011 (= IBI01 test 1), etc.
    fn = filename.upper()
    fn_match = re.match(r"^IBI(\d{2,3})", fn)
    if fn_match:
        digits = fn_match.group(1)
        # IBI011 -> IBI01 (3-digit: first 2 are machine, last is test number)
        if len(digits) == 3 and digits[0] == "0":
            return f"IBI{digits[:2]}"
        # IBI01, IBI13, IBI14, IBI15, IBI16 -> as-is (2 digits)
        return f"IBI{digits[:2]}"

    return "unknown"


def workload_overlap(kws_a: list[str], kws_b: list[str]) -> float:
    """Score how similar two workload keyword sets are. 0-1."""
    if not kws_a or not kws_b:
        return 0.0
    # Remove "clone" from comparison — it's a deployment detail, not a workload
    a = set(kws_a) - {"clone"}
    b = set(kws_b) - {"clone"}
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


# ─── Comparison logic ─────────────────────────────────────────────────

def find_comparison_groups(captures: list[CaptureInfo]) -> list[dict]:
    """Find groups of captures that can be meaningfully compared.

    Groups are formed by:
    1. No-GPU baseline + GPU captures on the same machine (original logic)
    2. Same hostname across multiple captures — even if all have GPU data,
       different profiles/configs on the same host are comparable

    A comparison group needs:
    - At least two captures from the same machine group
    - Ideally different GPU configs (no-GPU vs GPU, or different profiles)
    """
    # First pass: group all captures by machine group (hostname)
    by_machine: dict[str, list[CaptureInfo]] = {}
    for c in captures:
        mg = extract_machine_group(c.name, c.hostname)
        if mg and mg != "unknown":
            by_machine.setdefault(mg, []).append(c)

    groups = []

    for machine_group, caps in by_machine.items():
        if len(caps) < 2:
            continue

        no_gpu = [c for c in caps if not c.has_gpu_data]
        with_gpu = [c for c in caps if c.has_gpu_data]

        if no_gpu:
            # Original logic: use no-GPU as baseline, compare against GPU captures
            for baseline in no_gpu:
                gpu_matches = []
                for gpu_cap in with_gpu:
                    overlap = workload_overlap(baseline.workload_keywords,
                                              gpu_cap.workload_keywords)
                    gpu_matches.append(gpu_cap)

                if gpu_matches:
                    groups.append({
                        "baseline": baseline,
                        "gpu_captures": gpu_matches,
                        "machine_group": machine_group,
                    })
        elif len(with_gpu) > 1:
            # All captures have GPU — compare different profiles/configs
            # Use the capture with the smallest profile as "baseline"
            from sizing_data import VGPU_PROFILES
            def profile_fb(c):
                return VGPU_PROFILES.get(c.vgpu_profile, 0)

            sorted_caps = sorted(with_gpu, key=profile_fb)
            baseline = sorted_caps[0]
            others = sorted_caps[1:]

            # Only create group if there are actually different configs
            profiles = set(c.vgpu_profile or c.gpu_model for c in with_gpu)
            if len(profiles) > 1:
                groups.append({
                    "baseline": baseline,
                    "gpu_captures": others,
                    "machine_group": machine_group,
                })

    return groups


def format_metric_comparison(baseline: CaptureInfo, gpu_caps: list[CaptureInfo],
                             metric_key: str, label: str, unit: str,
                             lower_is_better: bool = False) -> list[str]:
    """Format a comparison row for one metric across baseline + GPU captures."""
    bl_stat = baseline.metrics.get(metric_key)
    if not bl_stat or bl_stat.samples == 0:
        return []

    lines = []
    bl_val = bl_stat.p95_val if "util" in metric_key.lower() or metric_key in ("cpu_util", "ram_pct") else bl_stat.avg_val

    for cap in gpu_caps:
        cap_stat = cap.metrics.get(metric_key)
        if not cap_stat or cap_stat.samples == 0:
            continue
        cap_val = cap_stat.p95_val if "util" in metric_key.lower() or metric_key in ("cpu_util", "ram_pct") else cap_stat.avg_val

        if bl_val > 0:
            delta = cap_val - bl_val
            delta_pct = (delta / bl_val) * 100
            direction = "lower" if delta < 0 else "higher"
            better = (delta < 0) == (not lower_is_better) if lower_is_better else (delta < 0)
            indicator = "improved" if ((delta < 0 and not lower_is_better) or (delta > 0 and lower_is_better)) else "worse" if ((delta > 0 and not lower_is_better) or (delta < 0 and lower_is_better)) else "same"
        else:
            delta_pct = 0
            indicator = "—"

    return lines


def generate_comparison_report(groups: list[dict], output_dir: Path | None = None) -> str:
    """Generate a markdown comparison report for all identified groups."""
    sections = [
        "# GPU vs No-GPU Comparison Report\n",
        "*Automated comparison of paired captures — same workload with and without GPU offload*\n",
        "---\n",
    ]

    if not groups:
        sections.append("No comparable GPU vs no-GPU pairs found in the dataset.\n")
        return "\n".join(sections)

    for i, group in enumerate(groups, 1):
        bl = group["baseline"]
        gpu_caps = group["gpu_captures"]
        machine = group["machine_group"]

        # Section header
        bl_workload = " + ".join(k for k in bl.workload_keywords if k != "clone") or "general"
        sections.append(f"\n## Comparison {i}: {bl_workload.title()} — {machine}\n")

        # Build comparison table header
        profile_cols = []
        for cap in gpu_caps:
            profile = cap.vgpu_profile or "GPU"
            fb = f" ({cap.gpu_model})" if cap.gpu_model and cap.gpu_model != "Unknown" else ""
            profile_cols.append(f"{cap.name}")

        header = f"| Metric | No GPU (Baseline) |"
        separator = "|--------|:-----------------:|"
        for col in profile_cols:
            header += f" {col} |"
            separator += ":-----------------:|"

        sections.append(header)
        sections.append(separator)

        # Metrics to compare
        comparisons = [
            ("cpu_util", "CPU Utilization (P95)", "%", True),
            ("ram_pct", "RAM Usage (P95)", "%", True),
            ("ram_used", "RAM Used (P95)", "MB", True),
            ("gpu_util", "GPU Utilization (P95)", "%", False),
            ("gpu_fb_used", "FB Used (Max)", "MB", False),
            ("gpu_fb_pct", "FB Usage (P95)", "%", False),
            ("nvenc", "NVENC Encode (P95)", "%", False),
            ("nvdec", "NVDEC Decode (P95)", "%", False),
            ("protocol_fps", "Protocol FPS (Avg)", "fps", False),
            ("protocol_rtt", "Protocol RTT (Avg)", "ms", True),
        ]

        for metric_key, label, unit, lower_better in comparisons:
            bl_stat = bl.metrics.get(metric_key)
            if not bl_stat or bl_stat.samples == 0:
                # Still show row if GPU captures have this metric
                has_any = any(
                    cap.metrics.get(metric_key) and cap.metrics[metric_key].samples > 0
                    for cap in gpu_caps
                )
                if not has_any:
                    continue
                bl_display = "N/A"
                bl_val = None
            else:
                if metric_key in ("protocol_fps", "protocol_rtt"):
                    bl_val = bl_stat.avg_val
                elif metric_key == "gpu_fb_used":
                    bl_val = bl_stat.max_val
                else:
                    bl_val = bl_stat.p95_val
                bl_display = f"{bl_val:.1f}{unit}"

            row = f"| {label} | {bl_display} |"

            for cap in gpu_caps:
                cap_stat = cap.metrics.get(metric_key)
                if not cap_stat or cap_stat.samples == 0:
                    row += " N/A |"
                    continue

                if metric_key in ("protocol_fps", "protocol_rtt"):
                    cap_val = cap_stat.avg_val
                elif metric_key == "gpu_fb_used":
                    cap_val = cap_stat.max_val
                else:
                    cap_val = cap_stat.p95_val

                cap_display = f"{cap_val:.1f}{unit}"

                # Delta annotation
                if bl_val is not None and bl_val > 0:
                    delta_pct = ((cap_val - bl_val) / bl_val) * 100
                    if abs(delta_pct) < 2:
                        annotation = ""
                    elif (delta_pct < 0 and lower_better) or (delta_pct > 0 and not lower_better):
                        annotation = f" ({delta_pct:+.0f}%)"
                    else:
                        # For FPS, higher is better (lower_better=False, so positive delta = good)
                        # For CPU/RTT, lower is better (lower_better=True, so negative delta = good)
                        if (delta_pct < 0 and not lower_better):
                            annotation = f" ({delta_pct:+.0f}%)"
                        else:
                            annotation = f" ({delta_pct:+.0f}%)"
                elif bl_val is None:
                    annotation = ""
                else:
                    annotation = ""

                row += f" {cap_display}{annotation} |"

            sections.append(row)

        # Analysis narrative
        sections.append(f"\n### Analysis\n")

        # CPU offload assessment
        bl_cpu = bl.metrics.get("cpu_util")
        if bl_cpu and bl_cpu.p95_val > 50:
            gpu_cpus = []
            for cap in gpu_caps:
                cap_cpu = cap.metrics.get("cpu_util")
                if cap_cpu and cap_cpu.samples > 0:
                    delta = cap_cpu.p95_val - bl_cpu.p95_val
                    gpu_cpus.append((cap, delta))

            if gpu_cpus:
                best = min(gpu_cpus, key=lambda x: x[1])
                if best[1] < -10:
                    sections.append(
                        f"- **CPU offload confirmed**: Baseline CPU P95 is {bl_cpu.p95_val:.0f}%. "
                        f"With GPU ({best[0].vgpu_profile or 'enabled'}), CPU drops to "
                        f"{best[0].metrics['cpu_util'].p95_val:.0f}% "
                        f"({best[1]:+.0f}pp). GPU is successfully offloading rendering/encode work.")
                elif best[1] > 10:
                    sections.append(
                        f"- **CPU increase with GPU**: Baseline CPU P95 is {bl_cpu.p95_val:.0f}%, "
                        f"but with GPU it rises to {best[0].metrics['cpu_util'].p95_val:.0f}%. "
                        f"This may indicate the GPU-enabled workflow is doing more work (e.g., "
                        f"Enscape real-time rendering that was disabled without GPU).")
                else:
                    sections.append(
                        f"- **CPU similar with/without GPU**: Baseline {bl_cpu.p95_val:.0f}% vs "
                        f"GPU {best[0].metrics['cpu_util'].p95_val:.0f}%. "
                        f"The workload may be CPU-bound regardless of GPU availability.")

        # Protocol FPS comparison
        # NOTE: Lower FPS with GPU does NOT necessarily mean contention.
        # Without GPU, the protocol may capture a simpler desktop at higher FPS.
        # With GPU + Enscape/rendering, more complex frames are generated — fewer
        # but richer frames is expected and often acceptable.
        bl_fps = bl.metrics.get("protocol_fps")
        if bl_fps and bl_fps.avg_val > 0:
            for cap in gpu_caps:
                cap_fps = cap.metrics.get("protocol_fps")
                if cap_fps and cap_fps.avg_val > 0:
                    delta = cap_fps.avg_val - bl_fps.avg_val
                    if abs(delta) > 2:
                        if delta > 0:
                            sections.append(
                                f"- **Protocol FPS improved**: {bl_fps.avg_val:.1f} -> "
                                f"{cap_fps.avg_val:.1f} fps with {cap.vgpu_profile or 'GPU'} "
                                f"— GPU encode offload improving frame delivery.")
                        else:
                            # Check if GPU workload is heavier (higher GPU util = more complex rendering)
                            cap_gpu = cap.metrics.get("gpu_util")
                            gpu_note = ""
                            if cap_gpu and cap_gpu.p95_val > 40:
                                gpu_note = (f" GPU is at {cap_gpu.p95_val:.0f}% P95 — the "
                                            "GPU-enabled session is rendering more complex content "
                                            "(e.g., real-time ray tracing) that wasn't possible without GPU.")
                            else:
                                gpu_note = " Monitor whether user experience is acceptable at this FPS."
                            sections.append(
                                f"- **Protocol FPS lower with GPU**: {bl_fps.avg_val:.1f} -> "
                                f"{cap_fps.avg_val:.1f} fps with {cap.vgpu_profile or 'GPU'}. "
                                f"This is expected when GPU enables heavier rendering workloads "
                                f"(richer frames, not fewer).{gpu_note}")

        # RAM comparison
        bl_ram = bl.metrics.get("ram_pct")
        if bl_ram and bl_ram.p95_val > 60:
            sections.append(
                f"- **RAM pressure**: Baseline at {bl_ram.p95_val:.0f}% P95. "
                f"{'Consider increasing VM RAM allocation.' if bl_ram.p95_val > 75 else 'Monitor during GPU-enabled tests.'}")

        # Profile sizing ladder
        if len(gpu_caps) > 1:
            sections.append(f"\n### Profile Sizing Ladder\n")
            profiles_seen = []
            for cap in sorted(gpu_caps, key=lambda c: c.vgpu_profile or ""):
                fb_stat = cap.metrics.get("gpu_fb_used")
                gpu_stat = cap.metrics.get("gpu_util")
                profile = cap.vgpu_profile or "?"
                fb_max = f"{fb_stat.max_val:.0f} MB" if fb_stat and fb_stat.max_val > 0 else "N/A"
                gpu_p95 = f"{gpu_stat.p95_val:.0f}%" if gpu_stat and gpu_stat.max_val > 0 else "N/A"
                profiles_seen.append(f"- **{profile}**: FB peak {fb_max}, GPU P95 {gpu_p95}")
            sections.extend(profiles_seen)

    # Save report
    report_text = "\n".join(sections)
    if output_dir:
        report_path = output_dir / "gpu_vs_nogpu_comparison.md"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"Comparison report saved: {report_path}", file=sys.stderr)

    return report_text


# ─── Main ─────────────────────────────────────────────────────────────

def load_captures(csv_dir: Path, config_json: Path | None = None) -> list[CaptureInfo]:
    """Load all CSV files and their configs."""
    configs = {}
    if config_json and config_json.exists():
        with open(config_json, "r") as f:
            configs = json.load(f)

    captures = []
    for csv_file in sorted(csv_dir.glob("*.csv")):
        name = csv_file.stem

        # Load config if available (from batch_process output or standalone JSON)
        cfg = configs.get(name, {})

        # Parse CSV to get metrics
        try:
            column_data, headers, rows = parse_csv(csv_file)
        except Exception as e:
            print(f"WARNING: Could not parse {csv_file.name}: {e}", file=sys.stderr)
            continue

        # Compute stats for each metric
        metrics = {}
        metric_names = {
            "cpu_util": ("CPU Utilization", "%"),
            "ram_pct": ("RAM %", "%"),
            "ram_used": ("RAM Used", "MB"),
            "gpu_util": ("GPU Utilization", "%"),
            "gpu_fb_used": ("FB Used", "MB"),
            "gpu_fb_pct": ("FB %", "%"),
            "gpu_fb_total": ("FB Total", "MB"),
            "nvenc": ("NVENC", "%"),
            "nvdec": ("NVDEC", "%"),
            "protocol_fps": ("Protocol FPS", "fps"),
            "protocol_rtt": ("Protocol RTT", "ms"),
        }
        for key, (label, unit) in metric_names.items():
            if key in column_data and column_data[key]:
                metrics[key] = compute_stats(column_data[key], label, unit)

        # Determine if GPU data exists
        has_gpu = False
        gpu_stat = metrics.get("gpu_util")
        fb_stat = metrics.get("gpu_fb_used")
        if (gpu_stat and gpu_stat.max_val > 0) or (fb_stat and fb_stat.max_val > 0):
            has_gpu = True

        capture = CaptureInfo(
            name=name,
            csv_path=csv_file,
            hostname=cfg.get("hostname", ""),
            gpu_model=cfg.get("gpu_model", ""),
            vgpu_profile=cfg.get("vgpu_profile", ""),
            gpu_type=cfg.get("gpu_type", ""),
            has_gpu_data=has_gpu,
            workload_keywords=extract_workload_keywords(name),
            metrics=metrics,
        )
        captures.append(capture)

    return captures


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Compare GPU vs no-GPU captures")
    parser.add_argument("--dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--configs", default=None,
                        help="JSON file with per-capture config (from batch_process)")
    parser.add_argument("--output", default=None, help="Output report path")
    args = parser.parse_args()

    csv_dir = Path(args.dir)
    config_json = Path(args.configs) if args.configs else None
    output_dir = csv_dir if not args.output else Path(args.output).parent

    captures = load_captures(csv_dir, config_json)
    print(f"Loaded {len(captures)} captures ({sum(1 for c in captures if c.has_gpu_data)} with GPU, "
          f"{sum(1 for c in captures if not c.has_gpu_data)} without)", file=sys.stderr)

    groups = find_comparison_groups(captures)
    print(f"Found {len(groups)} comparison group(s)", file=sys.stderr)

    report = generate_comparison_report(groups, output_dir)
    print(report)


if __name__ == "__main__":
    main()
