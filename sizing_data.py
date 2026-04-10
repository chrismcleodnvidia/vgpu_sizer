"""
Static sizing tables for vGPU deployment recommendations.
Based on NVIDIA vPC/vWS sizing guides and field-validated customer guidelines.
"""

# ─── Workload definitions ──────────────────────────────────────────────
# Each workload has per-session resource recommendations and classification hints.

WORKLOADS = {
    "knowledge_worker": {
        "label": "Knowledge Worker",
        "description": "Office, WebGL, teleconferencing, training videos",
        "vcpu": 2,
        "cpu_oversub": 5,
        "min_clock_ghz": 2.4,
        "preferred_clock_ghz": 2.9,
        "ram_gb": (6, 8),
        "fb_range_gb": (2, 2),
        "license": "vPC",
        "profile_series": "B",
        "recommended_gpus": ["A16", "L4"],
        "single_thread_sensitive": True,
        "notes": "Ask about RTX/Tensor core use.",
    },
    "healthcare": {
        "label": "Healthcare (EPIC EMR)",
        "description": "EPIC and similar EMR applications",
        "vcpu": (2, 4),
        "cpu_oversub": 5,
        "min_clock_ghz": 2.4,
        "preferred_clock_ghz": 2.9,
        "ram_gb": (6, 8),
        "fb_range_gb": (2, 4),
        "license": "vPC",
        "profile_series": "B",
        "recommended_gpus": ["A16", "L4", "L40"],
        "single_thread_sensitive": True,
        "notes": "Later EPIC versions may require 4 vCPU.",
    },
    "2d3d": {
        "label": "2D/3D Design",
        "description": "AutoCAD, USTN/Bentley suites, etc.",
        "vcpu": 4,
        "cpu_oversub": 3,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (16, 16),
        "fb_range_gb": (2, 4),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L4", "L40"],
        "single_thread_sensitive": True,
        "notes": "Single-threaded rendering common. Ask about RTX/Tensor core use.",
    },
    "engineering": {
        "label": "Engineering (CAD)",
        "description": "SolidWorks, CATIA, Siemens NX, Creo, etc.",
        "vcpu": 4,
        "cpu_oversub": 3,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (16, 32),
        "fb_range_gb": (2, 4),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L40"],
        "single_thread_sensitive": True,
        "notes": "Model rebuild is single-threaded. Ask about RTX/Tensor core use.",
    },
    "gis": {
        "label": "GIS",
        "description": "Esri ArcGIS / ArcGIS PRO",
        "vcpu": (4, 10),
        "cpu_oversub": 3,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (16, 16),
        "fb_range_gb": (2, 4),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L40", "L4"],
        "single_thread_sensitive": True,
        "notes": "Map rendering often single-threaded. Ask about RTX/Tensor core use.",
    },
    "energy": {
        "label": "Energy",
        "description": "Petrel, Landmark, Kingdom, etc.",
        "vcpu": 6,
        "cpu_oversub": 3,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (32, 32),
        "fb_range_gb": (3, 16),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L40"],
        "single_thread_sensitive": True,
        "notes": "High FB demand. Ask about RTX/Tensor core use.",
    },
    "rendering": {
        "label": "Rendering",
        "description": "Real-time or offline rendering: CLO3D, V-Ray, KeyShot, Blender, etc.",
        "vcpu": (4, 8),
        "cpu_oversub": 2,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (16, 32),
        "fb_range_gb": (4, 16),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L40"],
        "single_thread_sensitive": False,
        "notes": "RTX cores critical for ray tracing. Tensor cores used for AI denoising. Low oversub ratio (2:1) recommended due to sustained high CPU+GPU load.",
    },
    "omniverse": {
        "label": "NVIDIA Omniverse",
        "description": "Omniverse Create/USD Composer, Omniverse View, Isaac Sim, Digital Twins, etc.",
        "vcpu": (6, 10),
        "cpu_oversub": 2,
        "min_clock_ghz": 3.0,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (64, 128),
        "fb_range_gb": (12, 96),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["RTX PRO 6000 BSE", "L40"],
        "single_thread_sensitive": False,
        "rtx_required": True,
        "ram_rule": "2.5x GPU VRAM",
        "storage_per_user_gb": 1024,
        "min_network_gbps": 10,
        "notes": (
            "RTX GPU REQUIRED (RT + Tensor cores mandatory). "
            "System RAM must be >= 2.5x GPU frame buffer for optimal performance (ECC recommended). "
            "Render-heavy workflows need more CPU cores + larger VRAM. "
            "Simulation-heavy workflows need higher CPU clock + larger VRAM. "
            "Textures and geometry must fit entirely in VRAM or they will not render. "
            "Physics uses GPU resources -- ensure headroom beyond rendering needs. "
            "Multi-GPU rendering scaling is nonlinear. "
            "10 Gb/s networking minimum per node for Design Collaboration workflows. "
            "HIGHLY RECOMMENDED to run a POC -- GPU resource usage across simulation, rendering, "
            "and inferencing is not a straight-line definition."
        ),
        "vgpu_profiles": {
            "L40": {"min": "12Q", "recommended": "48Q", "density": "1-4 users"},
            "L4": {"min": "20Q", "recommended": "2x 20Q", "density": "1 user"},
            "RTX PRO 6000 BSE": {"min": "24Q", "recommended": "96Q", "density": "1-4 users"},
        },
    },
    "gpu_compute": {
        "label": "GPU Compute / Benchmark",
        "description": "GPU stress tests (FurMark, SPECviewperf), CUDA compute, AI inference",
        "vcpu": (4, 8),
        "cpu_oversub": 2,
        "min_clock_ghz": 2.4,
        "preferred_clock_ghz": 3.0,
        "ram_gb": (16, 32),
        "fb_range_gb": (4, 16),
        "license": "vWS",
        "profile_series": "Q",
        "recommended_gpus": ["L40"],
        "single_thread_sensitive": False,
        "notes": "Benchmark results show peak GPU capacity, not typical user load. Size based on actual application workload, not benchmark scores. If this is a stress test, use it to validate GPU headroom rather than to size the deployment.",
    },
}


# ─── vGPU profile definitions ──────────────────────────────────────────
# Maps profile name -> frame buffer size in GB.

VGPU_PROFILES = {
    # B-series (vPC) — virtual desktops, 1 user per vGPU, FRL 45 FPS
    "1B": 1,
    "2B": 2,
    "3B": 3,
    "4B": 4,
    "8B": 8,
    # Q-series (vWS) — virtual workstations, 1 user per vGPU, FRL 60 FPS
    "1Q": 1,
    "2Q": 2,
    "4Q": 4,
    "8Q": 8,
    "12Q": 12,
    "16Q": 16,
    "20Q": 20,
    "24Q": 24,
    "48Q": 48,
    "96Q": 96,
    # A-series (vApps) — virtual applications (RDSH/Citrix Virtual Apps),
    # FB is allocated per vGPU but shared across all user sessions on the
    # RDSH host. Console display limited to 1x 1280x1024; per-session
    # resolution is controlled by the remoting solution. FRL 60 FPS.
    "1A": 1,
    "2A": 2,
    "4A": 4,
    "8A": 8,
    "16A": 16,
    "24A": 24,
    "48A": 48,
}

# Profile series metadata
PROFILE_SERIES_INFO = {
    "B": {
        "license": "vPC",
        "label": "Virtual Desktop",
        "description": "1 user per vGPU VM",
        "frl_fps": 45,
    },
    "Q": {
        "license": "vWS",
        "label": "Virtual Workstation",
        "description": "1 user per vGPU VM, full Quadro/RTX feature set",
        "frl_fps": 60,
    },
    "A": {
        "license": "vApps",
        "label": "Virtual Applications",
        "description": (
            "Multi-session (RDSH/Citrix Virtual Apps). FB is allocated per vGPU "
            "but shared across all user sessions on the host. Console display is "
            "1x 1280x1024; per-session resolution is set by the remoting solution. "
            "Density is determined by concurrent sessions on the RDSH host, not by "
            "vGPU count."
        ),
        "frl_fps": 60,
    },
}


# ─── GPU hardware specs ────────────────────────────────────────────────
# total_fb_gb = total frame buffer on the physical GPU
# max_vgpu_instances = max simultaneous vGPU VMs the card supports

GPU_HARDWARE = {
    "A16": {
        "total_fb_gb": 64,  # 4x16GB GPUs per card
        "gpu_chips": 4,
        "fb_per_chip_gb": 16,
        "form_factor": "PCIe",
        "license_types": ["vPC", "vApps"],
        "profiles": {
            "1B": 64, "2B": 32, "3B": 20, "4B": 16, "8B": 8,
            "1Q": 64, "2Q": 32, "4Q": 16,
            "1A": 64, "2A": 32, "4A": 16, "8A": 8, "16A": 4,
        },
    },
    "L4": {
        "total_fb_gb": 24,
        "gpu_chips": 1,
        "fb_per_chip_gb": 24,
        "form_factor": "PCIe (LP)",
        "license_types": ["vPC", "vWS", "vApps"],
        "rtx": True,
        "profiles": {
            "1B": 24, "2B": 12, "4B": 6,
            "1Q": 24, "2Q": 12, "4Q": 6,
            "20Q": 1,
            "1A": 24, "2A": 12, "4A": 6,
        },
    },
    "L40": {
        "total_fb_gb": 48,
        "gpu_chips": 1,
        "fb_per_chip_gb": 48,
        "form_factor": "PCIe",
        "license_types": ["vPC", "vWS", "vApps"],
        "rtx": True,
        "profiles": {
            "1B": 48, "2B": 24, "4B": 12, "8B": 6,
            "1Q": 48, "2Q": 24, "4Q": 12, "8Q": 6, "12Q": 4, "16Q": 3, "48Q": 1,
            "1A": 48, "2A": 24, "4A": 12, "8A": 6, "16A": 3, "48A": 1,
        },
    },
    "L40S": {
        "total_fb_gb": 48,
        "gpu_chips": 1,
        "fb_per_chip_gb": 48,
        "form_factor": "PCIe",
        "license_types": ["vPC", "vWS", "vApps"],
        "rtx": True,
        "profiles": {
            "1B": 48, "2B": 24, "4B": 12, "8B": 6,
            "1Q": 48, "2Q": 24, "4Q": 12, "8Q": 6, "12Q": 4, "16Q": 3, "48Q": 1,
            "1A": 48, "2A": 24, "4A": 12, "8A": 6, "16A": 3, "48A": 1,
        },
    },
    "RTX PRO 6000 BSE": {
        "total_fb_gb": 96,
        "gpu_chips": 1,
        "fb_per_chip_gb": 96,
        "form_factor": "PCIe",
        "license_types": ["vWS"],
        "rtx": True,
        "profiles": {
            "4Q": 24, "8Q": 12, "12Q": 8, "16Q": 6, "24Q": 4, "48Q": 2, "96Q": 1,
        },
    },
}


# ─── Profile sizing thresholds ─────────────────────────────────────────
# peak_fb_with_headroom → recommended profile
# Headroom factor: 1.2x (20% above peak)

FB_HEADROOM_FACTOR = 1.2

def recommend_profile(peak_fb_gb: float, series: str, workload_key: str = "") -> str:
    """Given peak FB usage (GB) and profile series (B or Q), return recommended profile.
    Omniverse has specific minimum profile requirements regardless of measured FB."""
    target = peak_fb_gb * FB_HEADROOM_FACTOR

    # Omniverse has a hard minimum of 12Q regardless of measured FB
    if workload_key == "omniverse":
        thresholds = [(12, "12Q"), (20, "20Q"), (24, "24Q"), (48, "48Q"), (96, "96Q")]
        for size, profile in thresholds:
            if target <= size:
                return profile
        return thresholds[-1][1]

    if series == "B":
        thresholds = [(2, "2B"), (3, "3B"), (4, "4B"), (8, "8B")]
    elif series == "A":
        thresholds = [(1, "1A"), (2, "2A"), (4, "4A"), (8, "8A"),
                      (16, "16A"), (24, "24A"), (48, "48A")]
    else:
        thresholds = [(2, "2Q"), (4, "4Q"), (8, "8Q"), (12, "12Q"),
                      (16, "16Q"), (20, "20Q"), (24, "24Q"), (48, "48Q"), (96, "96Q")]
    for size, profile in thresholds:
        if target <= size:
            return profile
    return thresholds[-1][1]


def get_users_per_gpu(profile: str, gpu_model: str) -> int | None:
    """Return max concurrent users for a profile on a given GPU, or None if unsupported."""
    gpu = GPU_HARDWARE.get(gpu_model)
    if not gpu:
        return None
    return gpu["profiles"].get(profile)


def get_vcpu(workload_key: str) -> str:
    """Return vCPU recommendation as a display string."""
    v = WORKLOADS[workload_key]["vcpu"]
    if isinstance(v, tuple):
        return f"{v[0]}-{v[1]}"
    return str(v)


def get_ram(workload_key: str) -> str:
    """Return RAM recommendation as a display string."""
    r = WORKLOADS[workload_key]["ram_gb"]
    if r[0] == r[1]:
        return f"{r[0]} GB"
    return f"{r[0]}-{r[1]} GB"


# ─── Filename-based workload hints ────────────────────────────────────────

# Maps lowercase keywords found in filenames to workload keys.
# Order matters: first match wins, so put more specific terms first.
FILENAME_WORKLOAD_HINTS: list[tuple[list[str], str, str]] = [
    # Omniverse
    (["omniverse", "isaac sim", "isaac_sim", "isaacsim", "usd composer",
      "usd_composer", "omni.create", "digital twin"],
     "omniverse", "NVIDIA Omniverse application"),
    # Energy / O&G
    (["petrel", "landmark", "kingdom", "geoframe", "openworks", "jewelsuite"],
     "energy", "Energy / O&G application"),
    # Healthcare
    (["epic", "emr", "meditech", "cerner", "allscripts"],
     "healthcare", "Healthcare EMR application"),
    # GIS
    (["arcgis", "arc gis", "arc_gis", "qgis", "mapinfo", "global mapper"],
     "gis", "GIS application"),
    # Engineering CAD
    (["solidworks", "solid works", "catia", "siemens nx", "creo", "inventor",
      "ansys", "abaqus", "nastran", "simulia"],
     "engineering", "Engineering CAD application"),
    # 2D/3D design
    (["autocad", "auto cad", "revit", "bentley", "microstation", "sketchup",
      "vectorworks", "archicad"],
     "2d3d", "2D/3D design application"),
    # Rendering
    (["adobe", "premiere", "after effects", "aftereffects", "photoshop",
      "illustrator", "lightroom", "indesign", "substance",
      "blender", "v-ray", "vray", "keyshot", "key shot",
      "clo3d", "clo 3d", "marvelous designer", "cinema4d", "cinema 4d",
      "c4d", "houdini", "maya", "3dsmax", "3ds max", "unreal", "unity",
      "davinci", "resolve", "nuke", "fusion studio"],
     "rendering", "Rendering / creative application"),
    # GPU compute / benchmarks
    (["furmark", "specviewperf", "specview", "benchmark", "cuda", "mlperf",
      "geekbench"],
     "gpu_compute", "GPU benchmark / compute workload"),
    # Knowledge worker
    (["office", "teams", "zoom", "webex", "citrix receiver", "workspace app"],
     "knowledge_worker", "Knowledge worker / productivity application"),
]


def classify_from_filename(filename: str) -> tuple[str, str, float] | None:
    """
    Attempt to infer workload type from the filename.
    Returns (workload_key, reasoning, confidence) or None if no match.
    """
    lower = filename.lower()
    for keywords, workload_key, description in FILENAME_WORKLOAD_HINTS:
        for kw in keywords:
            if kw in lower:
                return (workload_key,
                        f"Filename contains '{kw}' — detected as {description}",
                        0.75)
    return None


# ─── Workload auto-classification heuristics ────────────────────────────

def classify_workload(avg_gpu_util: float, peak_fb_gb: float,
                      avg_nvenc: float, avg_cpu: float,
                      filename: str = "",
                      gpu_data_present: bool = True) -> tuple[str, str, float]:
    """
    Attempt to classify workload from metrics and optionally from filename.
    Returns (workload_key, reasoning, confidence 0-1).

    When gpu_data_present=False (no GPU metrics in capture), filename-based
    classification is the primary signal. Confidence is reduced since GPU
    metrics can't validate the classification.
    """
    # ── Check filename first — strong signal if a known app name is present ──
    if filename:
        hint = classify_from_filename(filename)
        if hint:
            if not gpu_data_present:
                wk, reason, conf = hint
                return (wk,
                        f"{reason}. This is a **CPU-only baseline** capture (no GPU "
                        "assigned). Useful for comparing CPU-only vs GPU-accelerated "
                        "performance. FB sizing requires a paired GPU-enabled capture.",
                        conf * 0.6)
            return hint

    # ── No GPU data and no filename match — can't classify reliably ──
    if not gpu_data_present:
        return ("knowledge_worker",
                "CPU-only baseline capture (no GPU assigned) with no application "
                "keywords in filename. Defaulting to Knowledge Worker. Pair with "
                "a GPU-enabled capture for comparison and FB sizing.",
                0.2)

    # ── GPU Compute / Benchmark: very high GPU, very small FB, low CPU ──
    # Stress tests like FurMark, SPECviewperf peg the GPU but use minimal FB
    if avg_gpu_util > 60 and peak_fb_gb < 1.0 and avg_cpu < 20:
        return ("gpu_compute",
                f"Very high GPU util ({avg_gpu_util:.0f}%) with minimal FB ({peak_fb_gb:.1f} GB) "
                f"and low CPU ({avg_cpu:.0f}%) — appears to be a GPU benchmark or compute workload",
                0.7)

    # ── Rendering: high CPU AND high GPU, often large FB ──
    # Real-time or offline rendering (CLO3D, V-Ray, Blender, Omniverse, KeyShot)
    # Key differentiator from Energy: rendering drives both CPU and GPU hard simultaneously
    if avg_gpu_util > 25 and avg_cpu > 50 and peak_fb_gb > 2.0:
        return ("rendering",
                f"High CPU ({avg_cpu:.0f}%) + GPU ({avg_gpu_util:.0f}%) with large FB "
                f"({peak_fb_gb:.1f} GB) — consistent with rendering workload",
                0.6)

    # Also catch rendering with moderate GPU but very high CPU + large FB
    if avg_cpu > 60 and peak_fb_gb > 4.0:
        return ("rendering",
                f"Very high CPU ({avg_cpu:.0f}%) with large FB ({peak_fb_gb:.1f} GB) — "
                f"consistent with CPU-heavy rendering (offline render, simulation)",
                0.5)

    # ── Energy: high FB is the strongest signal, but NOT high CPU ──
    # Energy apps (Petrel, Landmark) are GPU+FB heavy but typically moderate CPU
    if peak_fb_gb > 4.0 and avg_cpu < 50:
        return ("energy",
                f"High frame buffer usage ({peak_fb_gb:.1f} GB peak) with moderate CPU ({avg_cpu:.0f}%)",
                0.5)

    # ── Knowledge worker: low everything ──
    if avg_gpu_util < 20 and peak_fb_gb < 2.0 and avg_nvenc < 5:
        return ("knowledge_worker",
                f"Low GPU util ({avg_gpu_util:.0f}%), small FB ({peak_fb_gb:.1f} GB), minimal encode",
                0.7)

    # ── Healthcare: similar to KW but slightly higher FB possible ──
    if avg_gpu_util < 20 and peak_fb_gb < 4.0 and avg_nvenc < 5:
        return ("healthcare",
                f"Low GPU util ({avg_gpu_util:.0f}%), moderate FB ({peak_fb_gb:.1f} GB), minimal encode",
                0.4)

    # ── Engineering / 2D/3D: moderate-high GPU, moderate FB ──
    if avg_gpu_util >= 20 and peak_fb_gb <= 4.0:
        if avg_gpu_util >= 40:
            return ("engineering",
                    f"Moderate-high GPU util ({avg_gpu_util:.0f}%), FB {peak_fb_gb:.1f} GB",
                    0.5)
        return ("2d3d",
                f"Moderate GPU util ({avg_gpu_util:.0f}%), FB {peak_fb_gb:.1f} GB",
                0.5)

    # ── GIS: variable GPU, moderate FB ──
    if peak_fb_gb <= 4.0:
        return ("gis",
                f"Variable GPU util ({avg_gpu_util:.0f}%), FB {peak_fb_gb:.1f} GB",
                0.4)

    # ── Fallback ──
    return ("engineering",
            f"GPU util {avg_gpu_util:.0f}%, FB {peak_fb_gb:.1f} GB -- defaulting to Engineering",
            0.3)
