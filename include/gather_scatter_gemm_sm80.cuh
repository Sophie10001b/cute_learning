#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cute/tensor.hpp>

namespace {

// shared memory
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define SMEM_SIZE 167936
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#define SMEM_SIZE 233472
#else
#define SMEM_SIZE 102400
#endif

__forceinline__ uint32_t next_pow_of_2(uint32_t x) {
    if (x <= 0) return 0;
    return 1 << (32 - __builtin_clz(x));
}

// activation type
enum class ActivationType {
    Identity,
    ReLU,
    SiLU
};

struct GEMMParams {
    const void* A;
    const void* B;
    const void* Mask;
    const void* Index;
    void* C;
    uint32_t M;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline,
          uint32_t kHiddenSize, uint32_t kIntermediateSize, uint32_t kGroupSize, uint32_t kNumGroup, uint32_t kGroupIter,
          uint32_t kRowG2SPerCTA, bool kUsePDL, typename DType, ActivationType kActivation>
__global__ void gather_scatter_gemm_kernel(const __grid_constant__ GEMMParams params) {

}


template <uint32_t kHiddenSize, uint32_t kIntermediateSize, uint32_t kGroupSize, uint32_t kNumGroup, bool kUsePDL, typename DType, ActivationType kActivation>
struct GatherScatterGEMMKernelSM80 {

    template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline, uint32_t kGroupIter, uint32_t kRowG2SPerCTA>
    static constexpr auto impl = gather_scatter_gemm_kernel<kBM, kBN, kBK, kPipeline, kHiddenSize, kIntermediateSize, kGroupSize, kNumGroup, kGroupIter, kRowG2SPerCTA, kUsePDL, DType, kActivation>;

    template <uint32_t kBN, uint32_t kGroupIter>
    static auto dispatch_BM(uint32_t BM, uint32_t BK, uint32_t pipeline) {
        switch (BM) {
            case 16: return dispatch_BK<16, kBN, kGroupIter>(BK, pipeline);
            case 32: return dispatch_BK<32, kBN, kGroupIter>(BK, pipeline);
            case 64: return dispatch_BK<64, kBN, kGroupIter>(BK, pipeline);
            case 128: return dispatch_BK<128, kBN, kGroupIter>(BK, pipeline);
            default: return dispatch_BK<16, kBN, kGroupIter>(BK, pipeline);
        }
    }

    template <uint32_t kBM, uint32_t kBN, uint32_t kGroupIter>
    static auto dispatch_BK(uint32_t BK, uint32_t pipeline) {
        switch (BK) {
            case 64: return dispatch_pipeline<kBM, kBN, 64, kGroupIter, kThreadsPerBlock / (64 / 8)>(pipeline);
            case 128: return dispatch_pipeline<kBM, kBN, 128, kGroupIter, kThreadsPerBlock / (128 / 8)>(pipeline);
            default: return dispatch_pipeline<kBM, kBN, 64, kGroupIter, kThreadsPerBlock / (64 / 8)>(pipeline);
        }
    }

    template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kGroupIter, uint32_t kRowG2SPerCTA>
    static auto dispatch_pipeline(uint32_t pipeline) {
        switch (pipeline) {
            case 2: return impl<kBM, kBN, kBK, 2, kGroupIter, kRowG2SPerCTA>;
            case 3: return impl<kBM, kBN, kBK, 3, kGroupIter, kRowG2SPerCTA>;
            case 4: return impl<kBM, kBN, kBK, 4, kGroupIter, kRowG2SPerCTA>;
            default: return impl<kBM, kBN, kBK, 3, kGroupIter, kRowG2SPerCTA>;
        }
    }

    static auto kernel_dispatch(uint32_t M, float sparsity) {
        using namespace host;
        constexpr uint32_t N = kIntermediateSize;
        constexpr uint32_t K = kHiddenSize;
        constexpr uint32_t G = kGroupSize;

        uint32_t pipiline = 3;
        uint32_t BM = next_pow_of_2(uint32_t(M * sparsity));
        if (BM > M) BM >>= 1;
        BM = std::min(uint32_t(128), std::max(uint32_t(16), BM));

        RuntimeCheck(
            std::has_single_bit(kGroupSize) && kGroupSize >= 16,
            "kGroupSize must be power of 2 and >= 16"
        );
        constexpr uint32_t BN = std::min(uint32_t(128), std::max(uint32_t(16), next_pow_of_2(kGroupSize)));
        constexpr uint32_t GroupIter = kGroupSize / BN;

        uint32_t BK = 128;
        while ((BM * BK + BN * BK) * pipiline > (SMEM_SIZE / 2) && BK > 64) BK >>= 1;
        
        RuntimeCheck(
            (BM * BK + BN * BK) * pipiline <= SMEM_SIZE,
            "BM * BK + BN * BK * pipiline must be less than or equal to SMEM_SIZE"
        );

        // dispatch
        const uint32_t kRowG2SPerCTA = kThreadsPerBlock / (BK / 8);
        auto kernel = dispatch_BM<BN, GroupIter>(BM, BK, pipiline);

        const dim3 grid_size = {cute::ceil_div(M, BM), kNumGroup, GroupIter};
        const dim3 block_size = {kThreadsPerBlock, 1, 1};

        return std::tuple(kernel, grid_size, block_size);
    }

    static void run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView Index,
        const tvm::ffi::TensorView C,
        const float sparsity
    ) {
        using namespace host;
        RuntimeCheck(
            sparsity >= 0 && sparsity < 1,
            "sparsity must be in [0, 1)"
        );

        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"intermediate_size"};
        auto K = SymbolicSize{"hidden_size"};
        auto G = SymbolicSize{"group_size"};
        auto NG = SymbolicSize{"num_groups"};
        auto device = SymbolicDevice{};
        
        N.set_value(kIntermediateSize);
        K.set_value(kHiddenSize);
        G.set_value(kGroupSize);
        NG.set_value(kNumGroup);
        device.set_options<kDLCUDA>();

        // host-side checking
        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(A);
        
        TensorMatcher({N, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(B);
        
        TensorMatcher({M, N}) //
            .with_strides({N, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(C);
        
        TensorMatcher({NG, M}) //
            .with_strides({M, 1})
            .with_dtype<uint8_t>()
            .with_device(device)
            .verify(Mask);
        
        TensorMatcher({NG, M}) //
            .with_strides({M, 1})
            .with_dtype<uint8_t>()
            .with_device(device)
            .verify(Index);
        
        RuntimeCheck(
            sizeof(DType) == 2,
            "DType must be fp16 or bf16"
        );
        
        static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
        const auto num_tokens = static_cast<uint32_t>(M.unwrap());

        RuntimeCheck(
            kHiddenSize % 128 == 0 && kIntermediateSize % 128 == 0,
            "kHiddenSize and kIntermediateSize must be divisible by 128"
        );

        const auto params = GEMMParams{
            .A = A.data_ptr(),
            .B = B.data_ptr(),
            .Mask = Mask.data_ptr(),
            .Index = Index.data_ptr(),
            .C = C.data_ptr(),
            .M = num_tokens
        };

        auto [kernel, grid_size, block_size] = kernel_dispatch(num_tokens, sparsity);
        LaunchKernel(grid_size, block_size, device.unwrap())
            .enable_pdl(kUsePDL)(kernel, params);
    }
};

} // namespace