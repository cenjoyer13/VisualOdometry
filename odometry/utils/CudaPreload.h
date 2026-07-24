#pragma once

// Self-contained CUDA / cuDNN runtime-library bootstrap.
//
// Background:
//   The installed ONNX Runtime 1.18.1 CUDA execution provider plugin here
//   (`libonnxruntime_providers_cuda.so`) is a CUDA-12 / cuDNN-9 build --
//   confirmed via `ldd`, which lists NEEDED entries for libcudart.so.12,
//   libcublasLt.so.12, libcublas.so.12, libcufft.so.11, and libcudnn.so.9.
//   Its own RUNPATH (`/usr/local/cuda-12.2/lib`) doesn't exist on this
//   machine, so all of the above have to come from <repo>/cuda12_libs/
//   instead. The system path only provides CUDA-11 cublas/cudart/curand
//   (a mismatched major version the provider can't use) and no cuDNN at
//   all. cuDNN 9's module layout is graph/ops/cnn/adv/heuristic/engines_*
//   (cuDNN 8's ops_infer/cnn_infer/... families are also bundled and kept
//   preloaded for any other cudnn8-only consumer, but aren't what this
//   provider needs). Each of these libs lazily dlopens siblings by name,
//   so the directory has to be on the loader's search path.
//
// Without this helper, users had to remember
//   export LD_LIBRARY_PATH=<repo>/cuda12_libs:$LD_LIBRARY_PATH
// before every run. This class does the same thing programmatically, plus
// explicitly preloads the CUDA/cuDNN sub-modules so they're already mapped
// when ORT's CUDA EP goes looking for them.
class CudaPreload {
public:
    // Locate cuda12_libs/ relative to the running binary, prepend it to
    // LD_LIBRARY_PATH, and dlopen the cuDNN sub-modules with RTLD_GLOBAL.
    // Idempotent; safe to call multiple times. No-op + warning if the
    // directory can't be located.
    //
    // Must be invoked BEFORE any ONNX Runtime session is constructed.
    // `verbose` mirrors OdometryConfig::verbose; when true, prints which
    // libs were preloaded.
    static void init(bool verbose = false);
};
