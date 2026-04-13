#pragma once
// sglang jit plugin
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
// cute & cooperative groups
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace {

// shared memory
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define SMEM_SIZE 167936
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#define SMEM_SIZE 233472
#else
#define SMEM_SIZE 102400
#endif

constexpr uint32_t numWarps = 4;
constexpr uint32_t threadsPerCTA = numWarps * device::kWarpThreads;
constexpr uint32_t G2SRowPerWarp = 4;
constexpr uint32_t G2SRowPerCTA = G2SRowPerWarp * numWarps;
constexpr uint32_t G2SColPerCTA = threadsPerCTA / G2SRowPerCTA;
constexpr uint32_t MMARowPerWarp = 16;
namespace cg = cooperative_groups;

// layout for smem
template <typename DType, class LayoutA, class LayoutB>
struct SharedMemory {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutA>> A;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutB>> B;
};

// helper
template <uint32_t Bytes>
struct CopyWidthToType;
template <>
struct CopyWidthToType<4> {
    using type = cute::uint32_t;
};
template <>
struct CopyWidthToType<8> {
    using type = cute::uint64_t;
};
template <>
struct CopyWidthToType<16> {
    using type = cute::uint128_t;
};

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper {
    uint8_t execute_cta;
    uint8_t execute_g2s_warp[kG2SIter];
    uint8_t execute_mma_warp[kMMAIter];

    uint8_t rMask[kG2SIter];
    uint8_t rMask_mma[kMMAIter];
    uint32_t rIndex[kG2SIter];
};

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kG2SRowPerCTA, uint32_t kPipeline, typename DType>
struct MMAHelper {
    // smem layout
    using sALayout = decltype(cute::tile_to_shape(
        cute::GMMA::Layout_K_SW128_Atom<DType>{},
        cute::make_shape(cute::Int<kBM>{}, cute::Int<kBK>{}, cute::Int<kPipeline>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));
    using sBLayout = decltype(cute::tile_to_shape(
        cute::GMMA::Layout_K_SW128_Atom<DType>{},
        cute::make_shape(cute::Int<kBN>{}, cute::Int<kBK>{}, cute::Int<kPipeline>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));

    // atom
    static constexpr uint32_t g2s_copy_width = (kBK / G2SColPerCTA) * sizeof(DType);
    using g2s_copy_type = CopyWidthToType<g2s_copy_width>::type;
    using g2sA_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2s_copy_type>>;
    using s2rA_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using g2sB_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2s_copy_type>>;
    using s2rB_atom = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, DType>;
    using mma_atom = cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>;

    // tv layout g2sA
    using g2sA_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sA_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<kBK / G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sA_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(g2sA_thr_layout{}, g2sA_val_layout{}))));
    using g2sA_tiled_copy = decltype(cute::make_tiled_copy(
        g2sA_atom{}, g2sA_thr_layout{}, g2sA_val_layout{}
    ));

    // tv layout g2sB
    using g2sB_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sB_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<kBN / G2SRowPerCTA>{}, cute::Int<kBK / G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sB_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(g2sB_thr_layout{}, g2sB_val_layout{}))));
    using g2sB_tiled_copy = decltype(cute::make_tiled_copy(
        g2sB_atom{}, g2sB_thr_layout{}, g2sB_val_layout{}
    ));
};

constexpr uint32_t next_pow_of_2(uint32_t x) {
    if (x <= 0) return 0;
    return 1 << (32 - __builtin_clz(x));
}

struct GEMMParams {
    const void* A;
    const void* B;
    const void* Mask;
    const void* Index;
    void* D;
    uint32_t M;
};

// L2 swizzle
template <uint32_t kL2Group>
__device__ __forceinline__ uint2 l2_swizzle(
    const uint32_t bidx, const uint32_t bidy,
    const uint32_t bdimx, const uint32_t bdimy
) {
    if constexpr (kL2Group <= 1) {
        return make_uint2(bidx, bidy);
    } else {
        const uint32_t pid = bidy * bdimx + bidx;
        const uint32_t num_pid_in_group = kL2Group * bdimy;
        const uint32_t group_id = pid / num_pid_in_group;
        const uint32_t first_pid_m = group_id * kL2Group;
        const uint32_t group_size_m = std::min(bdimx - first_pid_m, kL2Group);
        const uint32_t pid_in_group = pid % num_pid_in_group;
        const uint32_t pid_m = first_pid_m + (pid_in_group % group_size_m);
        const uint32_t pid_n = pid_in_group / group_size_m;
        return make_uint2(pid_m, pid_n);
    }
}

template <
typename gATensor, typename gBTensor, typename sATensor, typename sBTensor,
typename pDsATensor, typename pDsBTensor, typename pSgBTensor,
typename ThrCopyA, typename TiledCopyA, typename TiledCopyB,
typename sAtATensor, typename sBtBTensor,
typename idxTensor, uint32_t kPipeline
>
__device__ __forceinline__ uint32_t load_AB(
    const gATensor& gA,
    const pDsATensor& pDsA, const pDsBTensor& pDsB, const pSgBTensor& pSgB, const idxTensor& rIndex,
    const ThrCopyA& g2sA_thr_copy, const TiledCopyA& g2sA_tiled_copy, const TiledCopyB& g2sB_tiled_copy,
    uint32_t consumer
) {
    using namespace cute;
    const uint32_t cur_stage = consumer % kPipeline;

    CUTE_UNROLL
    for (uint32_t i=0, j=threadIdx.x / G2SColPerCTA; i < size(rIndex); ++i, j+=G2SRowPerCTA) {
        auto gAtA = gA(make_coord(_, _), make_coord(rIndex(i), consumer));
        auto pSgA = g2sA_thr_copy.partition_S(gAtA);
        copy(g2sA_tiled_copy, pSgA(make_coord(_, _), _, 0), pDsA(make_coord(_, _), j, _, cur_stage));
    }
    copy(g2sB_tiled_copy, pSgB(make_coord(_, _), _, _, cur_stage), pDsB(make_coord(_, _), _, _, cur_stage));

    cute::cp_async_fence();
    return consumer + 1;
}

// main kernel func
template <
uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kL2Group, uint32_t kPipeline,
uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
typename kMMAHelper, typename DType,
uint32_t kG2SRowPerCTA, uint32_t, uint32_t kMMARow
>
__global__ void gather_scatter_gemm_kernel(const __grid_constant__ GEMMParams params) {
    using namespace cute;
    const auto& [A, B, Mask, Index, D, M] = params;
    constexpr uint32_t N = kN;
    constexpr uint32_t K = kK;
    constexpr uint32_t NG = kNG;
    constexpr uint32_t NGIter = kNGIter;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t BK = kBK;
    constexpr uint32_t Pipeline = kPipeline;

    // (cdiv(M, BM), cdiv(N, BN), SplitK)
    const uint32_t tidx = threadIdx.x;
    const uint32_t NBM = cutlass::ceil_div(M, BM);
    const uint32_t NBN = gridDim.x / NBM;
    const uint32_t SplitK = gridDim.z;
    const uint2 tile_idx = l2_swizzle<kL2Group>(
        blockIdx.x % NBM, blockIdx.x / NBM,
        NBM, NBN
    );
    const uint32_t bidx = tile_idx.x;
    const uint32_t bidy = tile_idx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    const uint32_t base_off_m = bidx * BM;
    const uint32_t base_off_n = bidy * BN;
    const uint32_t base_off_k = bidz * BK;
    const uint32_t MMARowPerCTA = MMARowPerWarp * kMMARow;

    const uint32_t thread_off_m = base_off_m + tidx / G2SColPerCTA;

    using sALayout = typename kMMAHelper::sALayout;
    using sBLayout = typename kMMAHelper::sBLayout;
    using g2sA_tv_tiler = typename kMMAHelper::g2sA_tv_tiler;
    using g2sB_tv_tiler = typename kMMAHelper::g2sB_tv_tiler;
    using g2sA_tiled_copy = typename kMMAHelper::g2sA_tiled_copy;
    using g2sB_tiled_copy = typename kMMAHelper::g2sB_tiled_copy;
    using mma_atom = typename kMMAHelper::mma_atom;

    auto tiled_mma = make_tiled_mma(mma_atom{});

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
    Tensor mD = make_tensor(
        make_gmem_ptr(static_cast<DType*>(D)),
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
    using SharedMemory = SharedMemory<DType, sALayout, sBLayout>;
    SharedMemory &smem = *reinterpret_cast<SharedMemory*>(shared_memory);

    Tensor sA = make_tensor(
        make_smem_ptr(smem.A.begin()),
        make_shape(Int<BM>{}, Int<BK>{}, Int<Pipeline>{}),
        make_stride(Int<BK>{}, _1{}, Int<BK * BM>{})
    );
    Tensor sB = make_tensor(
        make_smem_ptr(smem.B.begin()),
        make_shape(Int<BN>{}, Int<BK>{}, Int<Pipeline>{}),
        make_stride(Int<BK>{}, _1{}, Int<BK * BN>{})
    );

    //
    // load mask & index
    // assum the BN = 128, N = 1024, NG = 2,
    // for token skip, the bdimy = 8, NG = 1, NGIter = 8, with coord (bidy // NGIter, ...)
    // for block skip, the bdimy = 8, NG = 2, NGIter = 4, with coord (bidy // NGIter, ...)
    //
    constexpr uint32_t G2SIter = BM / G2SRowPerCTA;
    constexpr uint32_t MMAIter = BM / (MMARowPerWarp * kMMARow);
    SkipHelper<G2SIter, MMAIter> skip_helper;
    CUTE_UNROLL
    for (uint32_t i=0, off=thread_off_m; i < G2SIter; ++i, off+=G2SRowPerCTA) {
        skip_helper.rMask[i] = off < M ? mMask(make_coord(bidy / NGIter, off)) : 0;
        skip_helper.rIndex[i] = off < M ? mIndex(make_coord(bidy / NGIter, off)) : 0;
    }

    // check if early exit current block or warp's g2s load
    skip_helper.execute_cta = 1;
    cg::thread_block this_cta = cg::this_thread_block();
    auto this_cta_tile = cg::tiled_partition<threadsPerCTA>(this_cta);
    CUTE_UNROLL
    for (uint32_t i=0; i < G2SIter; ++i) {
        skip_helper.execute_cta &= skip_helper.rMask[i];
        skip_helper.execute_g2s_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask[i] > 0)));
    }
    skip_helper.execute_cta = cg::reduce(this_cta_tile, skip_helper.execute_cta, cg::bit_and<uint8_t>{});
    if (skip_helper.execute_cta == 0) return;

    // check if mma need execute
    CUTE_UNROLL
    for (uint32_t i=0, off=base_off_m + tidx / (threadsPerCTA / MMARowPerCTA); i < MMAIter; ++i, off+=MMARowPerCTA) {
        skip_helper.rMask_mma[i] = off < M ? mMask(make_coord(bidy / NGIter, off)) : 0;
        skip_helper.execute_mma_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask_mma[i] > 0)));
    }

    //
    // prologue, load A & B
    //
    uint8_t producer = 0;
    uint8_t consumer = 0;

    Tensor gA = zipped_divide(mA, g2sA_tv_tiler{});
    Tensor gB = local_tile(mB, g2sB_tv_tiler{}, make_coord(bidy, _));

    ThrCopy g2sA_thr_copy = g2sA_tiled_copy{}.get_slice(tidx % G2SColPerCTA);
    ThrCopy g2sB_thr_copy = g2sB_tiled_copy{}.get_slice(tidx);

    auto pDsA = g2sA_thr_copy.partition_D(sA);
    auto pDsB = g2sB_thr_copy.partition_D(sB);
    auto pSgB = g2sB_thr_copy.partition_S(gB);

    // fill the pipeline
    CUTE_UNROLL
    for (uint32_t i=0; i < Pipeline - 1; i++) {
        consumer = load_AB(gA, pDsA, pDsB, pSgB, skip_helper.rIndex, g2sA_thr_copy, g2sA_tiled_copy{}, g2sB_tiled_copy{}, consumer);
    }

    //
    // mainloop, issue s2r & mma
    //
    ThrMMA thr_mma = tiled_mma.get_slice(tidx);

}

template <
uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kSplitK, uint32_t kPipeline,
bool kUsePDL, typename DType, uint32_t kAct
>
struct GatherScatterGEMMKernel {
    static void run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView Index,
        const tvm::ffi::TensorView D,
        const float sparsity
    ) {
        using namespace host;
        RuntimeCheck(
            sparsity >= 0 && sparsity < 1,
            "sparsity must be in [0, 1)"
        );

        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"out_features"};
        auto K = SymbolicSize{"in_features"};
        auto NG = SymbolicSize{"num_groups"};
        auto NGIter = SymbolicSize{"num_groups_iter"};
        auto device = SymbolicDevice{};

        N.set_value(kN);
        K.set_value(kK);
        NG.set_value(kNG);
        NGIter.set_value(kNGIter);
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
            .verify(D);
        
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

        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        RuntimeCheck(
            kK % 32 == 0 && kN % 16 == 0,
            "N and K must be divisible by 16 and 32"
        );

        const auto params = GEMMParams{
            .A = A.data_ptr(),
            .B = B.data_ptr(),
            .Mask = Mask.data_ptr(),
            .Index = Index.data_ptr(),
            .D = D.data_ptr(),
            .M = num_tokens
        };

        // host-side static tiling
        MMAHelper<kBM, kBN, kBK, G2SRowPerCTA, kPipeline, DType> mma_helper;
        using sALayout = decltype(mma_helper.sALayout);
        using sBLayout = decltype(mma_helper.sBLayout);

        static constexpr size_t smem_size = size_t(sizeof(SharedMemory<DType, sALayout, sBLayout>));
        constexpr auto kernel = gather_scatter_gemm_kernel<
            kBM, kBN, kBK, 4, kPipeline, kN, kK, kNG, kNGIter,
            mma_helper, DType, G2SRowPerCTA, kBM / MMARowPerWarp
        >;

        const dim3 grid_size = {cute::ceil_div(params.M, kBM), cute::ceil_div(kN, kBN), kSplitK};
        const dim3 block_size = {threadsPerCTA, 1, 1};
        LaunchKernel(grid_size, block_size, device.unwrap(), smem_size).enable_pdl(kUsePDL)(kernel, params);
    }
};


} // namespace
