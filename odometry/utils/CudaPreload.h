#pragma once

// Self-contained CUDA / cuDNN runtime-library bootstrap.
//
// Background:
//   ONNX Runtime 1.18.1's CUDA execution provider plugin is built against
//   CUDA 11 and NEEDs libcudnn.so.8. The system path provides cublas/cudart
//   for CUDA 11 but not cuDNN; cuDNN is bundled in <repo>/cuda12_libs/.
//   libcudnn.so.8 itself lazily dlopens its sub-modules (cudnn_*_infer/train),
//   so the directory has to be on the loader's search path.
//
// Without this helper, users had to remember
//   export LD_LIBRARY_PATH=<repo>/cuda12_libs:$LD_LIBRARY_PATH
// before every run. This class does the same thing programmatically, plus
// explicitly preloads the cuDNN sub-modules so they're already mapped when
// libcudnn or ORT goes looking for them.
class CudaPreload {
public:
    // Locate cuda12_libs/ relative to the running binary, prepend it to
    // LD_LIBRARY_PATH, and dlopen the cuDNN sub-modules with RTLD_GLOBAL.
    // Idempotent; safe to call multiple times. No-op + warning if the
    // directory can't be located.
    //
    // Must be invoked BEFORE any ONNX Runtime session is constructed.
    // `verbose` mirrors OdometryConfig::verbose — when true, prints which
    // libs were preloaded.
    static void init(bool verbose = false);
};
