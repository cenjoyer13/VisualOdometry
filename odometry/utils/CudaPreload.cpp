#include "CudaPreload.h"

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

// cuDNN 8 sub-module order: ops_infer is the base; cnn_infer/adv_infer build
// on it; cudnn.so itself is the umbrella that lazily resolves all of them.
const char* kPreloadOrder[] = {
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
