#include "CudaPreload.h"

#ifndef __linux__
// Non-Linux platforms: ONNX Runtime resolves CUDA libs through the OS loader
// search (PATH on Windows, DYLD_LIBRARY_PATH on macOS) without needing the
// process-internal preload dance Linux requires for libcudnn's lazy sibling
// loads. init() is a no-op here.
void CudaPreload::init(bool /*verbose*/) {}
#else

#include <dlfcn.h>
#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::string resolveLibsDir() {
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';

    namespace fs = std::filesystem;
    fs::path exe(buf);
    // Layout: <repo>/build/<binary> -> cuda12_libs sits at <repo>/cuda12_libs.
    fs::path candidate = exe.parent_path() / ".." / "cuda12_libs";

    std::error_code ec;
    fs::path resolved = fs::weakly_canonical(candidate, ec);
    if (ec) return {};
    if (!fs::is_directory(resolved, ec) || ec) return {};
    return resolved.string();
}

void prependEnvPath(const char* var, const std::string& dir) {
    const char* existing = std::getenv(var);
    std::string updated = dir;
    if (existing && *existing) {
        updated.push_back(':');
        updated += existing;
    }
    setenv(var, updated.c_str(), /*overwrite=*/1);
}

// Order matters: each entry must come after whatever it depends on (ldd
// confirms these via `NEEDED` entries), so an eager dlopen never fails
// looking for a not-yet-mapped sibling.
//
// cudart/cublasLt/cublas/cufft: the actual installed ONNX Runtime CUDA EP
// here is a CUDA-12 build (see CudaPreload.h), and needs these from
// cuda12_libs/ specifically -- cublas depends on cublasLt, the rest are
// independent.
//
// cuDNN 9 sub-module order: graph is the base (everything else needs it);
// ops also needs graph; cnn/adv/heuristic/engines_* need graph (cnn/adv
// also need ops); cudnn.so itself is the umbrella that lazily resolves the
// rest by name.
const char* kPreloadOrder[] = {
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcufft.so.11",

    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn.so.9",

    // cuDNN 8 family: kept for any other CPU/cudnn8-only consumer; harmless
    // if unused by the CUDA-12 provider above.
    "libcudnn_ops_infer.so.8",
    "libcudnn_cnn_infer.so.8",
    "libcudnn_adv_infer.so.8",
    "libcudnn_ops_train.so.8",
    "libcudnn_cnn_train.so.8",
    "libcudnn_adv_train.so.8",
    "libcudnn.so.8",
};

}  // namespace

void CudaPreload::init(bool verbose) {
    static bool already_done = false;
    if (already_done) return;
    already_done = true;

    const std::string libs_dir = resolveLibsDir();
    if (libs_dir.empty()) {
        std::cerr << "[CudaPreload] cuda12_libs/ not found next to the binary; "
                     "ONNX CUDA EP will rely on the system loader (likely CPU fallback).\n";
        return;
    }

    // 1) Env path for lazy dlopens (libcudnn dispatches to its sub-modules
    //    by name, not by full path, so the directory has to be searchable).
    prependEnvPath("LD_LIBRARY_PATH", libs_dir);

    // 2) Explicit RTLD_GLOBAL preload guarantees the symbols are mapped even
    //    if some downstream loader opens with RTLD_LOCAL.
    int loaded = 0;
    for (const char* name : kPreloadOrder) {
        std::string full = libs_dir + "/" + name;
        void* h = dlopen(full.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            std::cerr << "[CudaPreload] dlopen failed: " << name
                      << " (" << dlerror() << ")\n";
            continue;
        }
        ++loaded;
        if (verbose) {
            std::cout << "[CudaPreload] loaded " << name << "\n";
        }
    }

    if (verbose) {
        std::cout << "[CudaPreload] dir=" << libs_dir
                  << " loaded=" << loaded << "/" << (sizeof(kPreloadOrder)/sizeof(*kPreloadOrder))
                  << "\n";
    }
}

#endif  // __linux__
