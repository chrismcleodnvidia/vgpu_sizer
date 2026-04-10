---
name: vgpu-sizer
description: "Analyze GPUProfiler output to size vGPU deployments. Supports 3 scenarios: baseline->POC, POC performance analysis, and troubleshooting. Accepts CSV, GPD files, or a directory of GPD files for batch processing. Usage: /vgpu-sizer [file or directory path]"
user_invocable: true
---

# vGPU Sizing Analyzer

You are helping size an NVIDIA vGPU deployment by analyzing GPUProfiler output data.

**IMPORTANT:** Replace `<VGPU_SIZER_DIR>` in all commands below with the actual path where you cloned this repository (e.g., `C:/tools/nvidia-vgpu-sizer`).

## Arguments

The user may pass a file path or directory path as an argument. If not provided, ask for it.

- **Single file** (`.csv` or `.gpd`): Analyze one capture (interactive, single-file workflow below)
- **Directory**: Batch-process all `.gpd` files in the directory (automated batch workflow below)

## Step 1: Gather Inputs

Ask the user for:

1. **File path** — path to a `.csv` or `.gpd` file from GPUProfiler (if not provided as argument)
2. **Scenario** — which analysis to perform:
   - **1 — Baseline capture**: Physical workstation profiled -> generate POC starting recommendations
   - **2 — POC analysis**: Evaluate current vGPU config performance
   - **3 — Troubleshooting**: Diagnose a reported issue
3. **Workload type** (optional) — if the user provides context about the application (e.g., "AutoCAD", "Office apps", "Petrel"), map it:
   - Office / WebGL / teleconferencing / training -> `knowledge_worker`
   - EPIC / EMR -> `healthcare`
   - AutoCAD / Bentley / Revit -> `2d3d`
   - SolidWorks / CATIA / NX / Creo -> `engineering`
   - ArcGIS / ArcGIS PRO -> `gis`
   - Petrel / Landmark / Kingdom -> `energy`
   - CLO3D / V-Ray / KeyShot / Blender / real-time or offline rendering -> `rendering`
   - NVIDIA Omniverse (Create, USD Composer, View, Isaac Sim, Digital Twins) -> `omniverse`
   - FurMark / SPECviewperf / CUDA compute / AI inference benchmarks -> `gpu_compute`
   - If unclear, omit — the script will auto-classify and you confirm with the user

**Scenario-specific inputs:**
- Scenario 2: ask for `--current-profile` ONLY if no GPD screenshot is available (the screenshot contains the vGPU profile)
- Scenario 3: ask for a brief symptom description (e.g., "user reports lag during CAD model rotation")

## Step 2: Handle File Type

### If CSV file:
Proceed directly to Step 3.

### If GPD file (single file):
GPD is a binary format. The CSV doesn't contain system config, but the GPD does — visible in GPUProfiler's GUI. The `batch_process.py` script handles the full pipeline automatically (config extraction + CSV export + analysis) for single files too.

Run the batch processor on the single file's parent directory with a filename filter, or use the manual workflow:

**Automated (preferred):**
```bash
python <VGPU_SIZER_DIR>/batch_process.py \
  --dir "<directory_containing_gpd>" \
  --scenario <1|2|3|auto> \
  --output-dir "<output_directory>"
```
This will process all GPD files in the directory. If you only need one file, use the manual workflow below.

**Manual single-file workflow:**

**2a. Launch GPUProfiler, extract config, and auto-export CSV:**
```bash
python <VGPU_SIZER_DIR>/screenshot_config.py \
  --gpd "<gpd_file_path>" \
  --no-close \
  --screenshot "<gpd_file_path_without_ext>_screenshot.png"
```
This opens GPUProfiler with the file, reads system config directly from the GUI controls via pywinauto, and outputs JSON to stdout. The `--no-close` flag keeps GPUProfiler open.

The JSON output contains fields like: `cpu_model`, `cpu_clock_ghz`, `gpu_model`, `gpu_memory_gb`, `driver`, `total_ram_gb`, `vgpu_profile`, `hostname`, `os`.

A `gpu_model` of `-NA-` means no GPU was detected in the session — this is a **valid result** (e.g., knowledge worker without GPU offload). Treat these as baseline (scenario 1) captures. Do NOT pass `-NA-` values as `--gpu-name` to analyze.py.

**2b. Export CSV via pywinauto:**
The Export button (automation_id=1028) opens a "Save As" child window. The filename edit is automation_id=1148, Save button is automation_id=1. Or ask the user to click Export and save manually.

**2c.** Once CSV exists, proceed to Step 3 with the config values from the JSON output.

## Batch Mode (Directory of GPD files)

When the user provides a **directory path** (or you detect multiple GPD files), use the batch processor:

```bash
python <VGPU_SIZER_DIR>/batch_process.py \
  --dir "<directory_containing_gpds>" \
  --scenario <1|2|3|auto> \
  [--output-dir "<output_directory>"] \
  [--workload <type>]
```

**Key flags:**
- `--scenario auto` (recommended): Auto-detects per file — vGPU configs (detected from GPU name like `GRID RTX6000P-4Q`) run as **Scenario 2 (POC)**, physical/no-GPU configs run as **Scenario 1 (Baseline)**
- `--output-dir`: Where to save CSVs and reports (defaults to same directory as GPDs)
- `--workload`: Force a workload type for all files, or omit for auto-classification per file

**What the batch processor does automatically:**
1. Opens each GPD in GPUProfiler
2. Extracts system config via pywinauto (hostname, CPU, GPU, RAM, driver, vGPU profile)
3. Exports CSV via the Save As dialog (Export button id=1028 → Save As child window)
4. Closes GPUProfiler
5. Runs `analyze.py` with extracted config overrides
6. Outputs a JSON summary to stdout with per-file results

**Handling `-NA-` GPU values:** Some captures show GPU model as `-NA-` — this is valid (e.g., knowledge worker session without GPU offload, or profiler running before GPU was assigned). These are always treated as **Scenario 1 (Baseline)** regardless of filename. Do NOT pass `-NA-` as `--gpu-name` to analyze.py.

**After batch completes**, present a consolidated summary table and cross-file analysis. Offer to re-run individual files with different settings if needed.

**GPU vs No-GPU comparison**: If the batch results contain both no-GPU (`-NA-`) and GPU-enabled captures, automatically run the comparison report:

```bash
python <VGPU_SIZER_DIR>/compare.py \
  --dir "<csv_directory>" \
  --configs "<csv_directory>/configs.json"
```

This pairs captures from the same machine, compares metrics side-by-side (CPU offload, RAM, FPS, GPU util, FB usage), and generates narrative analysis. The comparison report is saved as `gpu_vs_nogpu_comparison.md` in the output directory.

To generate the configs.json, save the batch_process.py JSON output's `results[].config` keyed by name.

## Step 3: Run Analysis (single file)

Build the command with any config overrides extracted from the GPD screenshot:

```bash
python <VGPU_SIZER_DIR>/analyze.py \
  --file "<csv_file_path>" \
  --scenario <1|2|3> \
  [--workload <type>] \
  [--current-profile <profile>] \
  [--density <N>] \
  [--symptom "<description>"] \
  [--cpu-model "<model>"] \
  [--cpu-clock <ghz>] \
  [--gpu-name "<gpu model>"] \
  [--vgpu-profile "<profile>"] \
  [--driver "<version>"] \
  [--total-ram <gb>] \
  [--display-info "<info>"]
```

Note: `--vgpu-profile` automatically serves as `--current-profile` for Scenario 2, so you don't need to ask the user for it if the screenshot provided it.

## Step 4: Review Auto-Classification

If the workload was auto-inferred (no `--workload` flag), the report will show the classification with a confidence level. If confidence is below 60%, ask the user to confirm or correct the workload type, then re-run if changed.

## Step 5: Present Results

Display the markdown report inline. The report is also automatically saved as `<filename>_report.md` in the same directory as the input file. Tell the user where it was saved.

## Key Sizing References

These are built into the analysis script but useful context if the user asks follow-up questions:

| Workload | vCPU | CPU Oversub | Min Clock | RAM | FB | License | GPU |
|---|---|---|---|---|---|---|---|
| Knowledge Worker | 2 | 5:1 | 2.4 GHz | 6-8 GB | 2 GB | vPC | A16, L4 |
| Healthcare (EPIC) | 2-4 | 5:1 | 2.4 GHz | 6-8 GB | 2-4 GB | vPC | A16, L4, L40 |
| 2D/3D Design | 4 | 3:1 | 3.0 GHz | 16 GB | 2-4 GB | vWS | L4, L40 |
| Engineering (CAD) | 4 | 3:1 | 3.0 GHz | 16-32 GB | 2-4 GB | vWS | L40 |
| GIS | 4-10 | 3:1 | 3.0 GHz | 16+ GB | 2-4 GB | vWS | L40, L4 |
| Energy | 6 | 3:1 | 3.0 GHz | 32+ GB | 3-16 GB | vWS | L40 |
| Rendering | 4-8 | 2:1 | 3.0 GHz | 16-32 GB | 4-16 GB | vWS | L40 |
| Omniverse | 6-10 | 2:1 | 3.0 GHz | 64-128 GB | 12-96 GB | vWS | RTX PRO 6000 BSE, L40 |
| GPU Compute/Benchmark | 4-8 | 2:1 | 2.4 GHz | 16-32 GB | 4-16 GB | vWS | L40 |

**CPU clock speed warning**: Low aggregate CPU utilization does NOT mean the CPU is adequate for single-threaded applications (CAD, GIS, desktop apps). Always check host CPU per-core boost clock against the workload minimum.

## Follow-up Guidance

After presenting the report, offer to:
- Adjust workload type and re-run if classification seems wrong
- Run a different scenario on the same data
- Explain any specific metric or recommendation in detail
- Calculate server-level sizing for a specific user count
- For batch runs: produce a consolidated cross-file comparison or re-run specific files with different parameters
- Compare no-GPU vs GPU captures to quantify the benefit of GPU offload (e.g., encoding, rendering)
