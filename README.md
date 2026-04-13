# vGPU Sizer

> **Disclaimer:** This is an **unofficial** community tool, not an official NVIDIA product. It is not developed, maintained, or endorsed by NVIDIA. Sizing recommendations are best-effort guidance and should be validated against your specific environment. Use at your own risk.

A [Claude Code](https://claude.ai/claude-code) skill that analyzes [NVIDIA GPUProfiler](https://developer.nvidia.com/gpu-profiler) output to generate vGPU sizing recommendations.

## What It Does

Takes GPUProfiler capture data (`.gpd` or `.csv`) and produces detailed vGPU deployment recommendations including:

- **Profile selection** (e.g., `GRID L40-4Q`, `GRID A16-2B`) based on workload analysis
- **Server density** calculations with CPU, RAM, and GPU constraints
- **License type** guidance (vPC vs vWS)
- **Performance assessment** against workload-specific thresholds

### Supported Scenarios

| Scenario | Input | Output |
|---|---|---|
| **1 - Baseline** | Physical workstation capture | POC starting configuration |
| **2 - POC Analysis** | vGPU session capture | Performance evaluation + tuning |
| **3 - Troubleshooting** | Any capture + symptom | Root cause analysis + fixes |

### Supported Workloads

Knowledge Worker, Healthcare (EPIC), 2D/3D Design, Engineering (CAD), GIS, Energy, Rendering, NVIDIA Omniverse, GPU Compute. Auto-classification is available if the workload type isn't specified.

## Prerequisites

- **Windows** (GPD file processing requires GPUProfiler GUI automation)
- **Python 3.10+**
- **NVIDIA GPUProfiler** installed (for `.gpd` file processing)
- **pywinauto** (`pip install pywinauto`) - required only for `.gpd` file processing; CSV-only analysis has no external dependencies
- **Claude Code** CLI or desktop app

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-org>/nvidia-vgpu-sizer.git
   cd nvidia-vgpu-sizer
   ```

2. Install the Claude Code skill by copying the skill definition to your Claude Code skills directory:
   ```bash
   # Windows
   xcopy /E /I .claude\skills\vgpu-sizer %USERPROFILE%\.claude\skills\vgpu-sizer

   # Linux/macOS
   cp -r .claude/skills/vgpu-sizer ~/.claude/skills/vgpu-sizer
   ```

3. Edit the skill file (`~/.claude/skills/vgpu-sizer/skill.md`) and replace all occurrences of `<VGPU_SIZER_DIR>` with the full path to where you cloned this repo. For example:
   ```
   python C:/tools/nvidia-vgpu-sizer/batch_process.py \
   ```

4. Install Python dependency (only needed for GPD files):
   ```bash
   pip install pywinauto
   ```

## Usage

In Claude Code, invoke the skill:

```
/vgpu-sizer path/to/capture.gpd
/vgpu-sizer path/to/capture.csv
/vgpu-sizer path/to/directory/of/gpd/files
```

Claude will walk you through scenario selection and produce a sizing report.

### Standalone CLI Usage

The Python scripts also work standalone without Claude Code:

```bash
# Single CSV analysis
python analyze.py --file capture.csv --scenario 1

# Batch process a directory of GPD files
python batch_process.py --dir ./captures --scenario auto --output-dir ./reports

# Compare GPU vs no-GPU captures
python compare.py --dir ./reports --configs ./reports/configs.json
```

Run any script with `--help` for full option details.

## Files

| File | Description |
|---|---|
| `analyze.py` | Core analysis engine - parses CSV data, computes metrics, generates sizing reports |
| `batch_process.py` | Batch processor - automates GPD open/extract/export/analyze cycle |
| `compare.py` | Comparison report generator - GPU vs no-GPU side-by-side analysis |
| `sizing_data.py` | Reference data - vGPU profiles, workload thresholds, GPU specs |
| `screenshot_config.py` | GPUProfiler GUI automation - extracts system config from GPD files via pywinauto |
| `.claude/skills/vgpu-sizer/skill.md` | Claude Code skill definition |

## Disclaimer

This tool is an **unofficial** personal project. It is **not** an official NVIDIA product or service. NVIDIA does not endorse, support, or guarantee the accuracy of the recommendations produced by this tool. Always validate sizing recommendations through proper POC testing in your target environment.
