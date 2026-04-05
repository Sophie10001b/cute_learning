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
    uint32_t rIndex[kG2SIter];
};

template <uint32_t kBK, uint32_t kG2SRowPerCTA, typename DType>
struct MMAHelper {
    // gmem -> smem
    static constexpr uint32_t g2s_col_per_cta = threadsPerCTA / kG2SRowPerCTA;
    static constexpr uint32_t copy_width = (kBK / g2s_col_per_cta) * 2;
    using copy_type = CopyWidthToType<copy_width>::type;
    using g2sA_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<copy_type>>;
    using s2rA_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using g2sB_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<copy_type>>;
    using s2rB_atom = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, DType>;

    using mma_atom = cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>;
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
__device__ auto l2_swizzle(
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

// main kernel func
template <
uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kL2Group, uint32_t kPipeline,
uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
class SLayoutA, class SLayoutB, typename DType,
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

    const uint32_t base_off_m = bidx * BM;
    const uint32_t base_off_n = bidy * BN;
    const uint32_t base_off_k = bidz * BK;

    const uint32_t thread_off_m = base_off_m + tidx / G2SColPerCTA;

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
    using SharedMemory = SharedMemory<DType, SLayoutA, SLayoutB>;
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
    constexpr uint32_t MMAIter = BM / (16 * kMMARow);
    SkipHelper<G2SIter, MMAIter> skip_helper;
    CUTE_UNROLL
    for (uint32_t i=0, off=thread_off_m; i < G2SIter; ++i, off += G2SRowPerCTA) {
        skip_helper.rMask[i] = off < M ? mMask(make_coord(bidy / NGIter, off)) : 0;
        skip_helper.rIndex[i] = off < M ? mIndex(make_coord(bidy / NGIter, off)) : 0;
    }

    // check if early exit current block
    skip_helper.execute_cta = 1;
    cg::thread_block this_cta = cg::this_thread_block();
    auto this_cta_tile = cg::tiled_partition<threadsPerCTA>(this_cta);
    CUTE_UNROLL
    for (uint32_t i=0; i < G2SIter; ++i) skip_helper.execute_cta &= skip_helper.rMask[i];
    skip_helper.execute_cta = cg::reduce(this_cta_tile, skip_helper.execute_cta, cg::bit_and<uint8_t>{});
    if (skip_helper.execute_cta == 0) return;

    //
    // prologue, load A & B
    //
    uint8_t producer = 0;
    uint8_t consumer = 0;

}

} // namespace
