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
    uint32_t M;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <uint32_t kRowPerCTA, uint32_t kColPerCTA, int64_t kHiddenSize, bool kUsePDL, typename DType>
__global__ void __launch_bounds__(128) swiglu_kernel(const __grid_constant__ SwiGLUParams params) {
    using namespace cute;
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;

    const auto& [up, gate, output, M] = params;
    constexpr int64_t K = kHiddenSize;

    // 1. cute make layout
    const auto gA_layout = make_ordered_layout(make_shape(_, Int<K>{}), make_step(_1{}, _0{}));

    Tensor gA = make_tensor(make_gmem_ptr(reinterpret_cast<DType*>(up)), gA_layout);
    Tensor gB = make_tensor(make_gmem_ptr(reinterpret_cast<DType*>(gate)), gA_layout);
    Tensor gC = make_tensor(make_gmem_ptr(reinterpret_cast<DType*>(output)), gA_layout);

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
        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(up).verify(gate).verify(output);
        
        static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
        uint32_t row_per_cta = 8;
        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        while ((num_tokens / row_per_cta) < kNumSM * 2 && row_per_cta > 1) row_per_cta >>= 1;
        
        const auto params = SwiGLUParams{
            .up = up.data_ptr(),
            .gate = gate.data_ptr(),
            .output = output.data_ptr(),
            .M = num_tokens,
        };

        constexpr uint32_t col_per_cta = 16 / sizeof(DType);
        
        auto kernel = swiglu_kernel<1, col_per_cta, kHiddenSize, kUsePDL, DType>;
        switch (row_per_cta) {
            case 8:
                kernel = swiglu_kernel<8, col_per_cta, kHiddenSize, kUsePDL, DType>; break;
            case 4:
                kernel = swiglu_kernel<4, col_per_cta, kHiddenSize, kUsePDL, DType>; break;
            case 2:
                kernel = swiglu_kernel<2, col_per_cta, kHiddenSize, kUsePDL, DType>; break;
        }

        LaunchKernel(div_ceil(num_tokens, row_per_cta), kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel, params);
    }
};


} // namespace
