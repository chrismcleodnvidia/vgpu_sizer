#!/usr/bin/env python3
"""
Batch-process all GPD files in a directory:
  1. Open each in GPUProfiler
  2. Extract system config via pywinauto
  3. Auto-export CSV via pywinauto (click Export button + save dialog)
  4. Close GPUProfiler
  5. Run analyze.py on each CSV

Usage:
  python batch_process.py --dir "path/to/gpd/folder" --scenario 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import from existing scripts
sys.path.insert(0, str(Path(__file__).parent))
from screenshot_config import find_gpuprofiler, extract_config_from_window


def export_csv_via_gui(pid: int, output_csv: Path, max_wait: float = 15.0) -> bool:
    """Click Export (id=1028) which opens a Save As dialog, then type path and Save."""
    from pywinauto import Application, Desktop
    from pywinauto.keyboard import send_keys

    try:
        app = Application(backend="uia").connect(process=pid, timeout=10)
        win = app.top_window()
        win.set_focus()
        time.sleep(0.5)

        # Click the Export button (automation_id=1028)
        export_btn = None
        for child in win.descendants():
            try:
                if child.element_info.automation_id == "1028":
                    export_btn = child
                    break
            except Exception:
                continue

        if export_btn is None:
            print(f"  WARNING: Could not find Export button", file=sys.stderr)
            return False

        print(f"  Clicking Export...", file=sys.stderr)
        export_btn.click_input()
        time.sleep(2)

        # Export opens a "Save As" child window inside GPUProfiler
        save_dlg = None
        for child in win.descendants():
            try:
                if child.window_text() == "Save As" and child.element_info.control_type == "Window":
                    save_dlg = child
                    break
            except Exception:
                continue

        if save_dlg is None:
            print(f"  WARNING: Save As dialog not found after Export click", file=sys.stderr)
            return False

        # Find the filename edit (automation_id=1148)
        fn_edit = None
        for child in save_dlg.descendants():
            try:
                if (child.element_info.automation_id == "1148"
                        and child.element_info.control_type == "Edit"):
                    fn_edit = child
                    break
            except Exception:
                continue

        if fn_edit is None:
            print(f"  WARNING: Filename edit not found in Save As dialog", file=sys.stderr)
            send_keys("{ESCAPE}")
            return False

        fn_edit.set_focus()
        time.sleep(0.2)
        fn_edit.set_text(str(output_csv))
        time.sleep(0.3)

        # Click Save button (automation_id=1 inside the Save As dialog)
        for child in save_dlg.descendants():
            try:
                if (child.element_info.automation_id == "1"
                        and child.element_info.control_type == "Button"):
                    child.click_input()
                    print(f"  Clicked Save", file=sys.stderr)
                    break
            except Exception:
                continue

        time.sleep(1.5)

        # Handle overwrite confirmation if it appears
        for child in win.descendants():
            try:
                txt = child.window_text().lower()
                if any(kw in txt for kw in ["confirm", "replace", "overwrite", "already exists"]):
                    send_keys("{ENTER}")
                    time.sleep(0.5)
                    break
            except Exception:
                continue

        if output_csv.exists() and output_csv.stat().st_size > 100:
            return True
        else:
            print(f"  WARNING: CSV file not created or too small", file=sys.stderr)
            return False

    except Exception as e:
        print(f"  WARNING: CSV export failed: {e}", file=sys.stderr)
        return False


def detect_scenario(config: dict, filename: str) -> int:
    """Auto-detect scenario from config and filename.
    Returns 2 if vGPU detected, 1 otherwise (baseline).
    GPU model of "-NA-" means no GPU was detected — treat as baseline."""
    # Config-based detection (most reliable)
    if config.get("gpu_type") == "vgpu" or config.get("vgpu_profile"):
        return 2

    # If GPU was not detected at all (-NA- or missing), this is a no-GPU
    # capture — always baseline regardless of filename hints.
    gpu = config.get("gpu_model", "")
    if not gpu or gpu == "-NA-":
        return 1

    # Filename heuristic: explicit FB sizes like "4GB", "6GB", "8GB" + "clone"
    fn = filename.lower()
    if "clone" in fn:
        return 2
    import re
    if re.search(r'\b[2-8]\s*gb\b', fn):
        return 2

    return 1


def process_single_gpd(gpd_path: Path, scenario: int | None, profiler_exe: str,
                       extra_args: list[str] = None, output_dir: Path | None = None) -> dict:
    """Process one GPD file. If scenario is None, auto-detect from config."""
    result = {
        "gpd": str(gpd_path),
        "name": gpd_path.stem,
        "config": {},
        "csv": None,
        "report": None,
        "error": None,
        "scenario_used": None,
    }

    # Kill any existing GPUProfiler
    subprocess.run(["taskkill", "/f", "/im", "GPUProfiler.exe"],
                    capture_output=True)
    time.sleep(1)

    # Launch GPUProfiler
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Processing: {gpd_path.name}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    try:
        proc = subprocess.Popen([profiler_exe, str(gpd_path)])
    except Exception as e:
        result["error"] = f"Failed to launch GPUProfiler: {e}"
        return result

    print(f"  Waiting for GPUProfiler to load...", file=sys.stderr)
    time.sleep(6)

    # Extract config
    try:
        config = extract_config_from_window(proc.pid)
        result["config"] = config
        print(f"  Config: {len(config)} fields extracted", file=sys.stderr)
        for k, v in config.items():
            if k not in ("window_title",):
                print(f"    {k}: {v}", file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: Config extraction failed: {e}", file=sys.stderr)

    # Auto-detect scenario if not forced
    if scenario is None:
        effective_scenario = detect_scenario(result["config"], gpd_path.stem)
        print(f"  Auto-detected scenario: {effective_scenario} "
              f"({'POC analysis (vGPU)' if effective_scenario == 2 else 'Baseline'})",
              file=sys.stderr)
    else:
        effective_scenario = scenario
    result["scenario_used"] = effective_scenario

    # Export CSV
    if output_dir:
        csv_path = output_dir / (gpd_path.stem + ".csv")
    else:
        csv_path = gpd_path.with_suffix(".csv")
    print(f"  Exporting CSV to: {csv_path.name}", file=sys.stderr)
    csv_ok = export_csv_via_gui(proc.pid, csv_path)

    if csv_ok:
        result["csv"] = str(csv_path)
        print(f"  CSV exported successfully", file=sys.stderr)
    else:
        print(f"  CSV export failed - will need manual export", file=sys.stderr)
        result["error"] = "CSV export automation failed"

    # Close GPUProfiler
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        subprocess.run(["taskkill", "/f", "/im", "GPUProfiler.exe"],
                        capture_output=True)
    time.sleep(1)

    # Run analysis if CSV exists
    if csv_path.exists():
        result["csv"] = str(csv_path)
        analyze_cmd = [
            sys.executable,
            str(Path(__file__).parent / "analyze.py"),
            "--file", str(csv_path),
            "--scenario", str(effective_scenario),
        ]

        # Add config overrides from GPD extraction (skip "-NA-" sentinel values)
        cfg = result["config"]
        def is_valid(v):
            return v and str(v).strip() not in ("-NA-", "")

        if is_valid(cfg.get("cpu_model")):
            analyze_cmd.extend(["--cpu-model", cfg["cpu_model"]])
        if cfg.get("cpu_clock_ghz"):
            analyze_cmd.extend(["--cpu-clock", str(cfg["cpu_clock_ghz"])])
        if is_valid(cfg.get("gpu_model")):
            analyze_cmd.extend(["--gpu-name", cfg["gpu_model"]])
        if is_valid(cfg.get("vgpu_profile")):
            analyze_cmd.extend(["--vgpu-profile", cfg["vgpu_profile"]])
            # For scenario 2, vgpu-profile also serves as current-profile
            if effective_scenario == 2:
                analyze_cmd.extend(["--current-profile", cfg["vgpu_profile"]])
        if is_valid(cfg.get("driver")):
            analyze_cmd.extend(["--driver", cfg["driver"]])
        if cfg.get("total_ram_gb"):
            analyze_cmd.extend(["--total-ram", str(cfg["total_ram_gb"])])
        if cfg.get("cpu_cores"):
            analyze_cmd.extend(["--vcpu-count", str(cfg["cpu_cores"])])

        if extra_args:
            analyze_cmd.extend(extra_args)

        print(f"  Running analysis...", file=sys.stderr)
        try:
            proc_result = subprocess.run(analyze_cmd, capture_output=True, text=True, timeout=60)
            if proc_result.returncode == 0:
                result["report"] = proc_result.stdout
                # Report is also auto-saved by analyze.py
                report_path = csv_path.with_name(csv_path.stem + "_report.md")
                if report_path.exists():
                    result["report_file"] = str(report_path)
                print(f"  Analysis complete", file=sys.stderr)
            else:
                result["error"] = f"Analysis failed: {proc_result.stderr}"
                print(f"  Analysis error: {proc_result.stderr}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            result["error"] = "Analysis timed out"
            print(f"  Analysis timed out", file=sys.stderr)

    return result


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Batch process GPD files")
    parser.add_argument("--dir", required=True, help="Directory containing GPD files")
    parser.add_argument("--scenario", default="auto",
                        help="1, 2, 3, or 'auto' (detect from config: vGPU->2, physical->1)")
    parser.add_argument("--workload", default=None, help="Override workload type for all files")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV and report output")
    args = parser.parse_args()

    gpd_dir = Path(args.dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if not gpd_dir.is_dir():
        print(f"ERROR: Not a directory: {gpd_dir}", file=sys.stderr)
        sys.exit(1)

    gpd_files = sorted(gpd_dir.glob("*.gpd"))
    if not gpd_files:
        print(f"ERROR: No GPD files found in {gpd_dir}", file=sys.stderr)
        sys.exit(1)

    profiler = find_gpuprofiler()
    if not profiler:
        print("ERROR: GPUProfiler.exe not found", file=sys.stderr)
        sys.exit(1)

    # Parse scenario: "auto" -> None, otherwise int
    if args.scenario == "auto":
        scenario_val = None
        print(f"Scenario: AUTO (detect per-file: vGPU->POC, physical->Baseline)", file=sys.stderr)
    else:
        scenario_val = int(args.scenario)
        print(f"Scenario: {scenario_val} (forced for all files)", file=sys.stderr)

    print(f"Found {len(gpd_files)} GPD files in {gpd_dir}", file=sys.stderr)
    print(f"GPUProfiler: {profiler}", file=sys.stderr)

    extra_args = []
    if args.workload:
        extra_args.extend(["--workload", args.workload])

    results = []
    for gpd in gpd_files:
        r = process_single_gpd(gpd, scenario_val, profiler, extra_args, output_dir)
        results.append(r)

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"BATCH PROCESSING COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    success = [r for r in results if r.get("report")]
    failed_csv = [r for r in results if not r.get("csv") or not Path(r["csv"]).exists()]
    failed_analysis = [r for r in results if r.get("csv") and not r.get("report")]

    print(f"  Total: {len(results)}", file=sys.stderr)
    print(f"  Success: {len(success)}", file=sys.stderr)
    print(f"  CSV export failed: {len(failed_csv)}", file=sys.stderr)
    print(f"  Analysis failed: {len(failed_analysis)}", file=sys.stderr)

    # Output JSON summary
    summary = {
        "total": len(results),
        "successful": len(success),
        "failed_csv": len(failed_csv),
        "failed_analysis": len(failed_analysis),
        "results": [
            {
                "name": r["name"],
                "scenario": r.get("scenario_used"),
                "csv": r.get("csv"),
                "report_file": r.get("report_file"),
                "config": r.get("config", {}),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    print(json.dumps(summary, indent=2))

    # Save configs.json for compare.py
    actual_output = output_dir or gpd_dir
    configs_path = actual_output / "configs.json"
    configs_dict = {r["name"]: r.get("config", {}) for r in results}
    configs_path.write_text(json.dumps(configs_dict, indent=2), encoding="utf-8")
    print(f"  Configs saved: {configs_path}", file=sys.stderr)

    # Auto-run comparison when:
    #   1. Both no-GPU (-NA-) and GPU captures exist (original logic), OR
    #   2. Same hostname appears across multiple captures (e.g. physical vs vGPU,
    #      or different profile sizes on the same host)
    successful = [r for r in results if r.get("report")]

    has_no_gpu = any(
        r.get("config", {}).get("gpu_model") in ("-NA-", "") or
        not r.get("config", {}).get("gpu_model")
        for r in successful
    )
    has_gpu = any(
        r.get("config", {}).get("gpu_type") == "vgpu" or
        r.get("config", {}).get("vgpu_profile")
        for r in successful
    )

    # Check for duplicate hostnames (same machine captured multiple times)
    hostnames = {}
    for r in successful:
        hn = r.get("config", {}).get("hostname", "").strip()
        if hn and hn not in ("-NA-", ""):
            hostnames.setdefault(hn, []).append(r["name"])
    has_same_host = any(len(names) > 1 for names in hostnames.values())

    should_compare = (has_no_gpu and has_gpu) or has_same_host

    if should_compare:
        reasons = []
        if has_no_gpu and has_gpu:
            reasons.append("GPU and no-GPU captures detected")
        if has_same_host:
            dupes = {hn: names for hn, names in hostnames.items() if len(names) > 1}
            reasons.append(f"same hostname across captures: "
                           f"{', '.join(f'{hn} ({len(n)} files)' for hn, n in dupes.items())}")
        print(f"\n  Auto-comparison triggered — {'; '.join(reasons)}...", file=sys.stderr)

        compare_cmd = [
            sys.executable,
            str(Path(__file__).parent / "compare.py"),
            "--dir", str(actual_output),
            "--configs", str(configs_path),
        ]
        try:
            cmp_result = subprocess.run(compare_cmd, capture_output=True, text=True, timeout=30)
            if cmp_result.returncode == 0:
                summary["comparison_report"] = str(actual_output / "gpu_vs_nogpu_comparison.md")
                print(f"  Comparison report saved", file=sys.stderr)
            else:
                print(f"  Comparison failed: {cmp_result.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"  Comparison failed: {e}", file=sys.stderr)

    # List files needing manual CSV export
    if failed_csv:
        print(f"\nFiles needing manual CSV export:", file=sys.stderr)
        for r in failed_csv:
            print(f"  - {r['name']}", file=sys.stderr)


if __name__ == "__main__":
    main()
