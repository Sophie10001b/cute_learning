#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cute/tensor.hpp>

namespace {

struct SwiGLUParams {
    const void* up;
    const void* gate;
    void* output;
    uint32_t num_tokens;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <uint32_t kRowPerCTA, int64_t kHiddenSize, bool kUsePDL, typename DType>
__global__ void __launch_bounds__(128) swiglu_kernel(const __grid_constant__ SwiGLUParams params) {

}

template <int64_t kHiddenSize, bool kUsePDL, typename DType>
struct SwiGLUKernel {

    static void run(
        const tvm::ffi::TensorView up,
        const tvm::ffi::TensorView gate,
        const tvm::ffi::TensorView output,
    ) {
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        K.set_value(kHiddenSize);
        device.set_options<kDLCUDA>();

        // checking on host
        TensorMatcher({M, K})
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(up);
        TensorMatcher({M, K})
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(gate);
        TensorMatcher({M, K})
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(output);
        
        static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
        uint32_t row_per_cta = 8;
        while ((M / row_per_cta) < kNumSM * 2 && row_per_cta > 1) row_per_cta >>= 1;
        
        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        const auto params = SwiGLUParams{
            .up = up.data_ptr(),
            .gate = gate.data_ptr(),
            .output = output.data_ptr(),
            .num_tokens = num_tokens,
        };

        LaunchKernel()

    }
};


} // namespace
