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

template <typename DType, class LayoutA, class LayoutB>
struct SharedMemory {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutA>> A;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutB>> B;
};

constexpr uint32_t next_pow_of_2(uint32_t x) {
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

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline, class SLayoutA, class SLayoutB,
          uint32_t kHiddenSize, uint32_t kIntermediateSize, uint32_t kGroupSize, uint32_t kNumGroup, uint32_t kGroupIter,
          uint32_t kRowG2SPerCTA, bool kUsePDL, typename DType, uint32_t kActivation>
__global__ void gather_scatter_gemm_kernel(const __grid_constant__ GEMMParams params) {
    using namespace cute;
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t bdimx = gridDim.x;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    constexpr auto BM = kBM;
    constexpr auto BN = kBN;
    constexpr auto BK = kBK;
    constexpr auto Pipeline = kPipeline;
    constexpr auto N = kIntermediateSize;
    constexpr auto K = kHiddenSize;
    constexpr auto G = kGroupSize;
    constexpr auto NG = kNumGroup;
    constexpr auto GI = kGroupIter;
    constexpr auto RowThreads = kRowG2SPerCTA;
    constexpr auto ColThreads = kThreadsPerBlock / RowThreads;
    constexpr auto BMLoop = BM / RowThreads;
    constexpr auto BNLoop = BN / RowThreads;
    constexpr auto BKLoop = BK / ColThreads;
    const auto& [A, B, Mask, Index, C, M] = params;
    constexpr uint32_t vec_width = 128 / sizeof(DType);

    //
    // prepare tensors and smem layout
    //

    Tensor mA = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(A)),
        make_shape(M, Int<K>{}),
        make_stride(Int<K>{}, _1{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(B)),
        make_shape(N, Int<K>{}),
        make_stride(Int<K>{}, _1{})
    );
    Tensor mC = make_tensor(
        make_gmem_ptr(static_cast<DType*>(C)),
        make_shape(M, Int<N>{}),
        make_stride(Int<N>{}, _1{})
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_shape(Int<NG>{}, M),
        make_stride(M, _1{})
    );
    Tensor mIndex = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(Index)),
        make_shape(Int<NG>{}, M),
        make_stride(M, _1{})
    );

    extern __shared__ uint8_t shared_memory[];
    using SharedMemory = SharedMemory<DType, SLayoutA, SLayoutB>;
    SharedMemory &smem = *reinterpret_cast<SharedMemory*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), make_shape(Int<BM>{}, Int<BK>{}, Int<Pipeline>{}));
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), make_shape(Int<BN>{}, Int<BK>{}, Int<Pipeline>{}));

    //
    // load mask & index
    //
    const uint32_t Mask_row_offset = bidx * BM;
    const uint32_t Mask_row = tidx / ColThreads;
    uint8_t rMask_arr[BMLoop];
    uint32_t rIndex_arr[BMLoop];
    uint8_t rMask_warp_arr[BMLoop];
    Tensor rMask = make_tensor(make_rmem_ptr(rMask_arr), make_shape(Int<BMLoop>{}));
    Tensor rIndex = make_tensor(make_rmem_ptr(rIndex_arr), make_shape(Int<BMLoop>{}));
    Tensor rMask_warp = make_tensor(make_rmem_ptr(rMask_warp_arr), make_shape(Int<BMLoop>{}));

    #pragma unroll
    for (uint32_t i=0, offset=Mask_row_offset + Mask_row; i < size(rMask); i++, offset+=RowThreads) {
        rMask(i) = offset < M ? mMask(make_coord(bidy, offset)) : 0;
        rIndex(i) = offset < M ? mIndex(make_coord(bidy, offset)) : 0;
        rMask_warp(i) = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(rMask(i))) > 0); // skip prologue under warp-level
    }

    // if (bidx + bidy + bidz == 0) print("[warp=%d, lane=%d], rMask=%d, rMask_warp=%d, rIndex=%d\n", warp_id, lane_id, rMask(0), rMask_warp(0), rIndex(0));

    //
    // load A & B
    //
    auto g2sA_thr = make_ordered_layout(make_shape(_1{}, Int<ColThreads>{}), make_step(_1{}, _0{}));
    auto g2sA_val = make_ordered_layout(make_shape(_1{}, Int<BKLoop>{}), make_step(_1{}, _0{}));
    auto g2sA_tv_tiler = product_each(shape(raked_product(g2sA_thr, g2sA_val)));
    TiledCopy g2sA_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, DType>{},
        g2sA_thr, g2sA_val
    );
    Tensor gA = zipped_divide(mA, g2sA_tv_tiler);

    auto g2sB_thr = make_ordered_layout(make_shape(Int<RowThreads>{}, Int<ColThreads>{}), make_step(_1{}, _0{}));
    auto g2sB_val = make_ordered_layout(make_shape(Int<BNLoop>{}, Int<BKLoop>{}), make_step(_1{}, _0{}));
    auto g2sB_tv_tiler = product_each(shape(raked_product(g2sB_thr, g2sB_val)));
    TiledCopy g2sB_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, DType>{},
        g2sB_thr, g2sB_val
    );
    Tensor gB = local_tile(mB, g2sB_tv_tiler, make_coord(bidy * GI + bidz, _));

    if (bidx + bidy + bidz == 0 && tidx == 0) {
        print(gA); print("\n");
        print(gB); print("\n");
    }
}


template <uint32_t kHiddenSize, uint32_t kIntermediateSize, uint32_t kGroupSize, uint32_t kNumGroup, bool kUsePDL, typename DType, uint32_t kActivation>
struct GatherScatterGEMMKernelSM80 {

    // template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline, uint32_t kGroupIter, uint32_t kRowG2SPerCTA>
    // static constexpr auto impl = gather_scatter_gemm_kernel<kBM, kBN, kBK, kPipeline, kHiddenSize, kIntermediateSize, kGroupSize, kNumGroup, kGroupIter, kRowG2SPerCTA, kUsePDL, DType, kActivation>;

    template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline, uint32_t kGroupIter, uint32_t kRowG2SPerCTA>
    static auto impl(const GEMMParams &params, const host::SymbolicDevice &device) {
        // smem layout dispatch
        using namespace cute;
        using SLayoutAAtom = decltype(GMMA::Layout_K_SW128_Atom<DType>{});
        using SLayoutBAtom = decltype(GMMA::Layout_K_SW128_Atom<DType>{});
        using SLayoutA = decltype(tile_to_shape(SLayoutAAtom{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kPipeline>{})));
        using SLayoutB = decltype(tile_to_shape(SLayoutBAtom{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<kPipeline>{})));

        static constexpr size_t smem_size = size_t(sizeof(SharedMemory<DType, SLayoutA, SLayoutB>));
        printf("Dynamic SMEM Size: %lu\n", smem_size);

        constexpr auto kernel = gather_scatter_gemm_kernel<kBM, kBN, kBK, kPipeline, SLayoutA, SLayoutB, kHiddenSize, kIntermediateSize, kGroupSize, kNumGroup, kGroupIter, kRowG2SPerCTA, kUsePDL, DType, kActivation>;

        const dim3 grid_size = {cute::ceil_div(params.M, kBM), kNumGroup, kGroupIter};
        const dim3 block_size = {kThreadsPerBlock, 1, 1};
        host::LaunchKernel(grid_size, block_size, device.unwrap(), smem_size).enable_pdl(kUsePDL)(kernel, params);
    }

    template <uint32_t kBN, uint32_t kGroupIter>
    static auto dispatch_BM(uint32_t BM, uint32_t BK, uint32_t pipeline, const GEMMParams &params, const host::SymbolicDevice &device) {
        switch (BM) {
            case 16: return dispatch_BK<16, kBN, kGroupIter>(BK, pipeline, params, device);
            case 32: return dispatch_BK<32, kBN, kGroupIter>(BK, pipeline, params, device);
            case 64: return dispatch_BK<64, kBN, kGroupIter>(BK, pipeline, params, device);
            case 128: return dispatch_BK<128, kBN, kGroupIter>(BK, pipeline, params, device);
            default: return dispatch_BK<16, kBN, kGroupIter>(BK, pipeline, params, device);
        }
    }

    template <uint32_t kBM, uint32_t kBN, uint32_t kGroupIter>
    static auto dispatch_BK(uint32_t BK, uint32_t pipeline, const GEMMParams &params, const host::SymbolicDevice &device) {
        switch (BK) {
            case 64: return dispatch_pipeline<kBM, kBN, 64, kGroupIter, kThreadsPerBlock / (64 / 8)>(pipeline, params, device);
            case 128: return dispatch_pipeline<kBM, kBN, 128, kGroupIter, kThreadsPerBlock / (128 / 8)>(pipeline, params, device);
            default: return dispatch_pipeline<kBM, kBN, 64, kGroupIter, kThreadsPerBlock / (64 / 8)>(pipeline, params, device);
        }
    }

    template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kGroupIter, uint32_t kRowG2SPerCTA>
    static auto dispatch_pipeline(uint32_t pipeline, const GEMMParams &params, const host::SymbolicDevice &device) {
        switch (pipeline) {
            case 2: return impl<kBM, kBN, kBK, 2, kGroupIter, kRowG2SPerCTA>(params, device);
            case 3: return impl<kBM, kBN, kBK, 3, kGroupIter, kRowG2SPerCTA>(params, device);
            case 4: return impl<kBM, kBN, kBK, 4, kGroupIter, kRowG2SPerCTA>(params, device);
            default: return impl<kBM, kBN, kBK, 3, kGroupIter, kRowG2SPerCTA>(params, device);
        }
    }

    static void kernel_dispatch(uint32_t M, float sparsity, const GEMMParams &params, const host::SymbolicDevice &device) {
        using namespace host;
        constexpr uint32_t N = kIntermediateSize;
        constexpr uint32_t K = kHiddenSize;
        constexpr uint32_t G = kGroupSize;

        uint32_t pipeline = 2;
        uint32_t BM = next_pow_of_2(uint32_t(M * sparsity));
        if (BM > M) BM >>= 1;
        BM = std::min(uint32_t(128), std::max(uint32_t(16), BM));

        RuntimeCheck(
            std::has_single_bit(G) && G >= 16,
            "kGroupSize must be power of 2 and >= 16"
        );
        constexpr uint32_t BN = std::min(uint32_t(128), std::max(uint32_t(16), next_pow_of_2(G)));
        constexpr uint32_t GroupIter = G / BN;

        uint32_t BK = 128;
        while ((BM * BK + BN * BK) * pipeline * sizeof(DType) > (SMEM_SIZE / 2) && BK > 64) BK >>= 1;
        while ((BM * BK + BN * BK) * pipeline * sizeof(DType) > (SMEM_SIZE / 2) && BM > 16) BM >>= 1;
        
        RuntimeCheck(
            (BM * BK + BN * BK) * pipeline <= SMEM_SIZE,
            "BM * BK + BN * BK * pipeline must be less than or equal to SMEM_SIZE"
        );

        const uint32_t kRowG2SPerCTA = kThreadsPerBlock / (BK / 8);

        // printf("BM: %d, BN: %d, BK: %d, pipeline: %d, GroupIter: %d, kRowG2SPerCTA: %d\n", BM, BN, BK, pipeline, GroupIter, kRowG2SPerCTA);
        dispatch_BM<BN, GroupIter>(BM, BK, pipeline, params, device);
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
            .with_dtype<uint32_t>()
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
        kernel_dispatch(num_tokens, sparsity, params, device);
    }
};

} // namespace