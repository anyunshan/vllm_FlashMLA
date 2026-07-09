#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>

namespace kerutils {

// Cached device properties (SM count, compute capability, ...) for the current
// device, queried via the CUDA Runtime and cached per device index. ABI-stable
// replacement for at::cuda::getCurrentDeviceProperties().
inline const cudaDeviceProp &get_cached_device_prop() {
    int device_idx = static_cast<int>(torch::stable::accelerator::getCurrentDeviceIndex());
    constexpr int kMaxDevices = 16;
    static cudaDeviceProp props[kMaxDevices];
    static bool inited[kMaxDevices] = {false};
    if (device_idx < 0 || device_idx >= kMaxDevices) {
        // Uncached fallback for unexpectedly large device indices.
        static thread_local cudaDeviceProp tmp;
        cudaGetDeviceProperties(&tmp, device_idx);
        return tmp;
    }
    if (!inited[device_idx]) {
        cudaGetDeviceProperties(&props[device_idx], device_idx);
        inited[device_idx] = true;
    }
    return props[device_idx];
}

}
