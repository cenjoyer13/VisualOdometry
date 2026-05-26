# VisualOdometry

A modular monocular visual-odometry pipeline in C++17. Supports classical
feature stacks (ORB, SIFT, FLANN, brute-force) and deep ones (SuperPoint,
ALIKED, LightGlue) via ONNX Runtime, with a GTSAM-based local bundle
adjustment running on a background thread. The project targets AirSim as
its primary host: it ships an interactive sandbox for live flight plus a
record-and-replay loop for reproducible evaluation against a known ground
truth.

---

## Build targets

| Target            | Platform     | Role                                                                                   |
| ----------------- | ------------ | -------------------------------------------------------------------------------------- |
| **FreeplayDrone** | Windows only | Interactive WASD flight in AirSim with on-demand odometry. Logs every frame to CSV.    |
| **PathRecorder**  | Windows only | WASD-fly in AirSim, save the ground-truth trajectory as a CSV (no odometry).           |
| **PathPlayer**    | Windows only | Replay a saved trajectory in AirSim and run odometry against the synthetic camera.     |
| **KittiEvaluator**| Linux + Win  | Replay a KITTI sequence through the pipeline. Legacy sanity-check target.              |

The three AirSim targets share the same `OdometryConfig` schema — the same
YAML config will drive Freeplay, Recorder, and Player runs.

---

## Requirements

Always:
- **C++17 toolchain** (GCC ≥ 9 / Clang ≥ 11 / MSVC 2022)
- **CMake ≥ 3.18**
- **OpenCV 4.8.0+** (without CUDA contrib — GPU inference goes through ONNX Runtime, not OpenCV)
- **ONNX Runtime 1.18.x** GPU build (download from
  [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases))
- **GTSAM 4.x**

Optional:
- **CUDA toolkit + cuDNN 8** for GPU inference. If you don't have these set
  up, just leave `system.backend: CPU` in your YAML — the pipeline runs end
  to end on CPU.
- **AirSim** (https://github.com/microsoft/AirSim) — required only for
  `FreeplayDrone` / `PathRecorder` / `PathPlayer`. Build `AirLib` and
  `rpclib` per AirSim's own instructions, then point the build at the repo
  root via `-DAIRSIM_ROOT=...`. `KittiEvaluator` builds without AirSim.

---

## Building on Linux

```bash
cmake --preset release-optimized
cmake --build build -j$(nproc)
```

This produces `build/KittiEvaluator`. The AirSim targets are Windows-only
and are skipped on Linux.

If ONNX Runtime is not at `/usr/local/onnxruntime`:
```bash
cmake --preset release-optimized -DONNXRUNTIME_DIR=/path/to/onnxruntime
```

## Building on Windows

Tested with Visual Studio 2022 (MSVC x64).

```bat
cmake --preset windows-msvc-release ^
    -DONNXRUNTIME_DIR=C:/onnxruntime-win-x64-gpu-1.18.1 ^
    -DAIRSIM_ROOT=C:/Path/To/AirSim
cmake --build build --preset windows-msvc-release
```

Binaries land under `build/Release/`. Make sure these DLLs are on `PATH` or
sitting beside the executables at run time:
- `onnxruntime.dll` (from `<ONNXRUNTIME_DIR>/lib/`)
- `onnxruntime_providers_cuda.dll` + `onnxruntime_providers_shared.dll`
  (same dir, only needed for CUDA)
- OpenCV runtime DLLs (`opencv_world*.dll` etc.)
- GTSAM runtime DLL
- AirSim's runtime DLLs (for the AirSim targets)

## Troubleshooting

**ONNX Runtime fails to load the CUDA execution provider.** This is almost
always a CUDA / cuDNN install mismatch (ONNX Runtime 1.18.x's CUDA EP needs
CUDA 11.x and cuDNN 8). The pipeline auto-falls back to CPU when the EP
fails, but you'll see the warning on stderr. If you don't need GPU
inference, drop CUDA support entirely by setting `system.backend: CPU` in
your YAML config. Everything works on CPU; deep detectors / matchers are
just slower.

---

## Running

Every target reads a YAML config. Examples live under `configs/`.

### FreeplayDrone (sandbox)

Start your AirSim sim, then:
```bat
FreeplayDrone.exe ..\configs\example_airsim.yaml --log my_flight.csv
```
Controls: WASD + arrow keys to fly, **Shift** for turbo, **O** to toggle
the odometry pipeline on/off, **Esc** to land and save the log.

Each frame is appended to `--log <path>` (default `freeplay_log.csv`) with
ground-truth and VO poses side-by-side, ready for postprocessing.

### Recorder → Player evaluation loop

Capture a flight:
```bat
PathRecorder.exe my_path.csv
```
Then replay it with odometry running against the same scene:
```bat
PathPlayer.exe my_path.csv ..\configs\example_airsim.yaml
```
PathPlayer writes `playback_log.csv` with time-aligned GT and VO columns.
Re-running the same `my_path.csv` against different YAML configs gives you
a fair A/B between detector / matcher / LBA settings.

### KittiEvaluator (legacy sanity check)

```bash
./build/KittiEvaluator configs/example_kitti.yaml
```
Useful for quick regressions on a public dataset when you don't have AirSim
running. Edit `dataset.root_path` in the YAML first.

### Global CLI flags

- `--debug` — force `system.verbose = true` for that run (LBA prints a
  correction-quality line per pass). Overrides the YAML value.

---

## Configuration

The YAML config has the following top-level blocks:

- **`camera`** — `fx, fy, cx, cy` for the source camera.
- **`detector`** — `type:` is one of `ORB`, `SIFT`, `SuperPoint`, `ALIKED`.
  Classical types take a sub-block with their parameters; deep types are
  configured at the model level.
- **`matcher`** — `type:` is one of `KinematicMatcher` (BF + ratio test,
  ORB-flavored), `FLANN`, `LightGlue`. `distance_ratio` controls the Lowe
  ratio test.
- **`bucketing`** — optional spatial filtering grid that keeps the top-N
  features per cell, so the feature distribution doesn't collapse into a
  texture-rich patch of the image.
- **`local_bundle_adjustment`** — GTSAM windowed BA on a background thread.
  Every numeric knob (window size, opt stride, noise sigmas, outlier
  thresholds, correction-publishing limits) is YAML-configurable. See
  `configs/example_airsim.yaml` for the keys.
- **`system`** — `backend` (`CPU` / `CUDA` / `OPENCL`), `num_threads`,
  `verbose`.
- **`dataset`** — KITTI-only (`root_path`, `sequence`).
- **`airsim`** — `playback_velocity` for PathPlayer.

The CMake-style FileStorage header `%YAML:1.0` on line 1 is **mandatory** —
OpenCV will refuse to open the file without it.

---

## Repository layout

```
.
├── CMakeLists.txt              cross-platform build (Linux + Windows)
├── CMakePresets.json           release-optimized (Linux) + windows-msvc-release
├── README.md
├── configs/
│   ├── example_airsim.yaml     starter config for the AirSim targets
│   └── example_kitti.yaml      starter config for KittiEvaluator
├── main.cpp                    FreeplayDrone entrypoint (Windows)
├── main_recorder.cpp           PathRecorder entrypoint (Windows)
├── main_player.cpp             PathPlayer entrypoint (Windows)
├── main_test.cpp               KittiEvaluator entrypoint (portable)
├── core/                       FreeplayDrone worker thread (Windows)
├── models/                     ONNX weights (SuperPoint, ALIKED, LightGlue)
└── odometry/                   pipeline + interfaces + factories
```
