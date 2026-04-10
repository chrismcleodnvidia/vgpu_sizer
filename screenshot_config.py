#!/usr/bin/env python3
"""
Launch GPUProfiler with a GPD file and extract system config from the GUI
using pywinauto text control reading. Also takes a screenshot as backup.

Outputs JSON with extracted system config to stdout.

Usage:
  python screenshot_config.py --gpd "path/to/file.gpd"
  python screenshot_config.py --gpd "path/to/file.gpd" --screenshot "output.png"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

GPUPROFILER_PATHS = [
    os.path.expanduser(r"~\Desktop\GPUProfiler.exe"),
    r"C:\internal tools\GPUProfiler.exe",
    r"C:\Program Files\GPUProfiler\GPUProfiler.exe",
]


def find_gpuprofiler() -> str | None:
    for p in GPUPROFILER_PATHS:
        if os.path.isfile(p):
            return p
    return None


def extract_config_from_window(pid: int) -> dict:
    """Read system config directly from GPUProfiler's text controls via pywinauto."""
    from pywinauto import Application

    app = Application(backend="uia").connect(process=pid, timeout=15)
    win = app.top_window()

    config = {}

    # Read all text children — GPUProfiler exposes config as Static text controls
    for child in win.children():
        try:
            text = child.window_text()
            if not text:
                continue

            # Parse known config patterns
            if text.startswith("Host:"):
                config["hostname"] = text.split(":", 1)[1].strip()
            elif text.startswith("OS:"):
                config["os"] = text.split(":", 1)[1].strip()
            elif text.startswith("CPU:"):
                # Format: "CPU:[20x @ 2.80GHz]" or "CPU: Intel Xeon..."
                cpu_text = text.split(":", 1)[1].strip().strip("[]")
                config["cpu_model"] = cpu_text
                clock_match = re.search(r"(\d+\.\d+)\s*GHz", cpu_text, re.I)
                if clock_match:
                    config["cpu_clock_ghz"] = float(clock_match.group(1))
                cores_match = re.search(r"(\d+)x", cpu_text)
                if cores_match:
                    config["cpu_cores"] = int(cores_match.group(1))
            elif text.startswith("Memory:"):
                mem_text = text.split(":", 1)[1].strip()
                mem_match = re.search(r"([\d.]+)\s*GB", mem_text, re.I)
                if mem_match:
                    config["total_ram_gb"] = float(mem_match.group(1))
            elif text.startswith("GPU:"):
                config["gpu_model"] = text.split(":", 1)[1].strip()
            elif text.startswith("GPU Memory:"):
                mem_text = text.split(":", 1)[1].strip()
                mem_match = re.search(r"([\d.]+)\s*GB", mem_text, re.I)
                if mem_match:
                    config["gpu_memory_gb"] = float(mem_match.group(1))
            elif text.startswith("Driver version:"):
                config["driver"] = text.split(":", 1)[1].strip()
            elif text.startswith("VBIOS:"):
                config["vbios"] = text.split(":", 1)[1].strip()
            elif text.startswith("License"):
                config["license"] = text.split(":", 1)[1].strip() if ":" in text else text

            # Detect vGPU profile from GPU name (e.g., "NVIDIA A10-4Q")
            if "gpu_model" in config and "vgpu_profile" not in config:
                gpu = config["gpu_model"]
                profile_match = re.search(r"-(\d+[BQCA])(?:\s|$)", gpu)
                if profile_match:
                    config["vgpu_profile"] = profile_match.group(1)

        except Exception:
            continue

    # Also capture the window title (contains filename)
    config["window_title"] = win.window_text()

    # Determine if this is a vGPU, passthrough, or physical GPU.
    # vGPU profiles have a profile suffix in the GPU name (e.g., "-4Q").
    # If no profile detected but we have GPU memory, it's a full GPU
    # (either physical workstation or GPU passthrough to a VM).
    if "vgpu_profile" not in config and "gpu_memory_gb" in config:
        config["gpu_type"] = "physical_or_passthrough"
        config["full_gpu_fb_gb"] = config["gpu_memory_gb"]
    elif "vgpu_profile" in config:
        config["gpu_type"] = "vgpu"

    return config


def take_screenshot(pid: int, output_path: str) -> bool:
    """Bring window to front and capture screenshot."""
    try:
        from pywinauto import Application
        from PIL import ImageGrab

        app = Application(backend="uia").connect(process=pid, timeout=5)
        win = app.top_window()
        win.maximize()
        win.set_focus()
        time.sleep(0.5)

        img = ImageGrab.grab()
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Screenshot failed: {e}", file=sys.stderr)
        return False


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Launch GPUProfiler and extract config")
    parser.add_argument("--gpd", required=True, help="Path to GPD file")
    parser.add_argument("--screenshot", default=None, help="Optional screenshot output path")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds to wait after launch (default: 5)")
    parser.add_argument("--no-close", action="store_true",
                        help="Leave GPUProfiler open (for manual CSV export)")
    args = parser.parse_args()

    gpd_path = Path(args.gpd)
    if not gpd_path.exists():
        print(f"ERROR: GPD file not found: {gpd_path}", file=sys.stderr)
        sys.exit(1)

    profiler = find_gpuprofiler()
    if not profiler:
        print("ERROR: GPUProfiler.exe not found.", file=sys.stderr)
        print("Searched:", file=sys.stderr)
        for p in GPUPROFILER_PATHS:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    # Kill any existing GPUProfiler
    subprocess.run(["taskkill", "/f", "/im", "GPUProfiler.exe"],
                    capture_output=True)
    time.sleep(1)

    # Launch GPUProfiler with the GPD file
    print(f"Launching: {profiler}", file=sys.stderr)
    print(f"Loading: {gpd_path.name}", file=sys.stderr)
    try:
        proc = subprocess.Popen([profiler, str(gpd_path)])
    except Exception as e:
        print(f"Failed to launch GPUProfiler: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Waiting {args.delay}s for GPUProfiler to load...", file=sys.stderr)
    time.sleep(args.delay)

    # Extract system config from text controls
    try:
        config = extract_config_from_window(proc.pid)
        print(f"Extracted {len(config)} config fields", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Could not read config from GUI: {e}", file=sys.stderr)
        config = {}

    # Optional screenshot
    if args.screenshot:
        screenshot_path = Path(args.screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        if take_screenshot(proc.pid, str(screenshot_path)):
            config["screenshot"] = str(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}", file=sys.stderr)

    # Close GPUProfiler unless --no-close
    if not args.no_close:
        try:
            proc.terminate()
            print("GPUProfiler closed.", file=sys.stderr)
        except Exception:
            pass
    else:
        print("GPUProfiler left open for manual CSV export.", file=sys.stderr)

    # Output config as JSON to stdout
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
