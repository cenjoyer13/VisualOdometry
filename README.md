# VisualOdometry

A modular monocular visual-odometry pipeline in C++17. Supports classical
feature stacks (ORB, SIFT, FLANN, brute-force) and deep ones (SuperPoint,
ALIKED, LightGlue) via ONNX Runtime, with a GTSAM-based local bundle
adjustment running on a background thread. Originally exercised on the
KITTI benchmark; extended with an AirSim recorder/player loop and a
free-flight sandbox.

## Build targets

| Target            | Platform     | Source              | Purpose                                                                       |
| ----------------- | ------------ | ------------------- | ----------------------------------------------------------------------------- |
| `KittiEvaluator`  | Linux + Win  | `main_test.cpp`     | Replay a KITTI sequence through the pipeline; emits `kitti_trajectory.csv`.   |
| `PathRecorder`    | Windows only | `main_recorder.cpp` | WASD-fly the drone in AirSim, log NED ground-truth pose to CSV at ~10 Hz.     |
| `PathPlayer`      | Windows only | `main_player.cpp`   | Replay a recorded CSV path in AirSim, run odometry against the sim camera.   |
| `FreeplayDrone`   | Windows only | `main.cpp`          | Live WASD flight + on-demand odometry + per-frame CSV log. Sandbox target.    |

All three AirSim-dependent targets share the same odometry pipeline and YAML
config schema — see `build/kitti_*.yaml` for examples.

## Requirements

Common (both OSes):
- **OpenCV 4.8.0+** (no CUDA contrib needed — GPU inference goes through ONNX Runtime)
- **ONNX Runtime 1.18.x GPU** (CUDA 11.x ABI). Get from
  https://github.com/microsoft/onnxruntime/releases
- **GTSAM 4.x** (system install)
- **CUDA 11.x toolkit + cuDNN 8** if you want GPU inference. On Linux the
  cuDNN 8 family lives in `cuda12_libs/` and is preloaded automatically; on
  Windows put cuDNN on `PATH` or beside the executable.

Windows-only (for `PathRecorder` / `PathPlayer` / `FreeplayDrone`):
- **AirSim** (https://github.com/microsoft/AirSim) — build AirLib + rpclib.

## Build — Linux

The `release-optimized` preset enables `-O3 -march=native -ffast-math` and
expects ONNX Runtime at `/usr/local/onnxruntime`.

```bash
cmake --preset release-optimized
cmake --build build -j$(nproc)
```

That gives you `build/KittiEvaluator`. AirSim targets are skipped on Linux —
they `#include <Windows.h>` and depend on Win32 APIs.

To use a non-default ONNX Runtime install path:
```bash
cmake -B build -DONNXRUNTIME_DIR=/opt/onnxruntime ...
```

### CUDA / cuDNN bundle on Linux
The system loader for CUDA on Linux can be quirky: ORT's CUDA EP `dlopen`s
`libcudnn.so.8`, which in turn lazily `dlopen`s its `cudnn_*_infer.so.8`
siblings *by name*. The repo's `cuda12_libs/` directory holds the cuDNN 8
family for this purpose. Place the libs in `cuda12_libs/` (already done in a
working clone) and `CudaPreload::init()` will discover them at startup
relative to the binary — no manual `LD_LIBRARY_PATH` export needed. If the
directory is empty or missing, ORT falls back to CPU. The CUDA-12 cudart /
cublas in `cuda12_libs/` are inert (ORT 1.18.1's CUDA EP plugin is built
against CUDA 11; system `/lib/x86_64-linux-gnu/` provides the rest).

## Build — Windows

Tested with Visual Studio 2022 + MSVC x64.

```bat
:: One-time setup — point CMake at your ONNX Runtime and AirSim installs.
cmake --preset windows-msvc-release ^
    -DONNXRUNTIME_DIR=C:/onnxruntime-win-x64-gpu-1.18.1 ^
    -DAIRSIM_ROOT=C:/Path/To/AirSim

cmake --build build --preset windows-msvc-release
```

Outputs land under `build/Release/`:
- `KittiEvaluator.exe`
- `PathRecorder.exe`
- `PathPlayer.exe`
- `FreeplayDrone.exe`

Runtime DLLs you typically need on PATH (or copied next to the binaries):
- `onnxruntime.dll` from `<ONNXRUNTIME_DIR>/lib/`
- `onnxruntime_providers_cuda.dll` + `onnxruntime_providers_shared.dll` (same dir)
- CUDA 11.x runtime + cuDNN 8 if you want GPU inference
- OpenCV + GTSAM DLLs

## Running

Each target reads a YAML config (sample configs live in `build/`):
```bash
./build/KittiEvaluator build/kitti_sift_flannmatch.yaml
./build/KittiEvaluator build/kitti_sift_lightgluematch.yaml --debug
```

The `--debug` flag forces `system.verbose = true` for that run, enabling
LBA's per-iteration log line.

YAML schema covers detector / matcher / bucketing / local-bundle-adjustment /
system / camera blocks. The `camera:` block (`fx, fy, cx, cy`) is honored
across all targets; without it KittiEvaluator falls back to KITTI Seq 00
intrinsics and AirSim targets fall back to AirSim 90 deg defaults.

## Repository layout

```
.
├── CMakeLists.txt              cross-platform build (Win/Linux)
├── CMakePresets.json           release-optimized (Linux) + windows-msvc-release
├── main_test.cpp               KittiEvaluator entrypoint (portable)
├── main_recorder.cpp           PathRecorder entrypoint (Win32)
├── main_player.cpp             PathPlayer entrypoint (Win32)
├── main.cpp                    FreeplayDrone entrypoint (Win32)
├── core/                       FreeplayDrone worker thread (Win32)
├── models/                     ONNX weights (SuperPoint, ALIKED, LightGlue)
├── cuda12_libs/                cuDNN 8 family + (unused) CUDA 12 cudart/cublas
└── odometry/                   pipeline + interfaces + factories
```

`CLAUDE.md` documents the internal architecture and known issues — useful
reading for anyone working on the pipeline itself.
