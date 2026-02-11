#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cute/tensor.hpp>

namespace {

// atomic add impl
template <typename DType, uint32_t NPack>
__device__ __forceinline__ void atomic_add_ptx(DType* addr, DType* val) {}

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300)
template <>
__device__ __forceinline__ void atomic_add_ptx<float, 1>(float* addr, float* val) {
    float __val = *val;
    float __tmp;
    asm volatile (
        "atom.global.relaxed.add.f32 %0, [%1], %2;\n"
        : "=f"(__tmp)
        : "l"(addr), "f"(__val)
    );
}
#endif

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500)
template <>
__device__ __forceinline__ void atomic_add_ptx<__half, 2>(__half* addr, __half* val) {
    uint32_t* __addr = reinterpret_cast<uint32_t*>(addr);
    __half2 __valh = __halves2half2(*val, *(val + 1));
    uint32_t __val = *reinterpret_cast<uint32_t*>(&__valh);
    uint32_t __tmp;
    asm volatile (
        "atom.global.relaxed.add.noftz.f16x2 %0, [%1], %2;\n"
        : "=r"(__tmp)
        : "l"(__addr), "r"(__val)
    );
}
#endif

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
template <>
__device__ __forceinline__ void atomic_add_ptx<__nv_bfloat16, 2>(__nv_bfloat16* addr, __nv_bfloat16* val) {
    uint32_t* __addr = reinterpret_cast<uint32_t*>(addr);
    __nv_bfloat162 __valh = __halves2bfloat162(*val, *(val + 1));
    uint32_t __val = *reinterpret_cast<uint32_t*>(&__valh);
    uint32_t __tmp;
    asm volatile (
        "atom.global.relaxed.add.noftz.bf16x2 %0, [%1], %2;\n"
        : "=r"(__tmp)
        : "l"(__addr), "r"(__val)
    );
}
#endif

// activation type
enum class ActivationType {
    Identity,
    ReLU,
    SiLU
};

// arch type
enum class ArchType {
    SM80,
    SM90,
    SM100
};

struct GatherScatterParams {
    const void* A;
    const void* IGather;
    const void* IScatter;
    void* B;
    uint32_t T;
    uint32_t M;
    uint32_t N;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <uint32_t kRowPerCTA, uint32_t kColPerCTA, uint32_t kHiddenSize, bool kUsePDL, typename DType>
__global__ void gather_scatter_kernel(const __grid_constant__ GatherScatterParams params) {
    using namespace cute;
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t lane_id = tidx % device::kWarpThreads;
    const uint32_t warp_id = tidx / device::kWarpThreads;

    // permute row loop, which allow handle boundary first
    const uint32_t bdimx = gridDim.x;
    const uint32_t bidx_perm = bdimx - bidx - 1;
    const uint32_t warp_row_offset = bidx_perm * kRowPerCTA + warp_id % kRowPerCTA;

    constexpr auto K = kHiddenSize;
    const auto& [A, IGather, IScatter, B, T, M, N] = params;
    constexpr uint32_t vec_width = 16 / sizeof(DType);
    const bool row_oob = warp_row_offset >= M;
    constexpr uint32_t atomic_add_width = 4 / sizeof(DType);

    Tensor mA = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(A)),
        make_shape(T, Int<K>{}),
        make_stride(Int<K>{}, Int<1>{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(static_cast<DType*>(B)),
        make_shape(N, Int<K>{}),
        make_stride(Int<K>{}, Int<1>{})
    );

    Tensor mIGather = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(IGather)),
        make_shape(M),
        make_stride(Int<1>{})
    );

    Tensor mIScatter = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(IScatter)),
        make_shape(M),
        make_stride(Int<1>{})
    );

    // 1. make tv layout
    auto thr_layout = make_ordered_layout(Shape<_1, Int<kColPerCTA>>{}, Step<_1, _0>{});
    auto val_layout = make_ordered_layout(Shape<_1, Int<vec_width>>{}, Step<_1, _0>{});

    // 2. make tiling for each row
    auto row_tiler = product_each(shape(raked_product(thr_layout, val_layout)));
    auto gA_row = zipped_divide(mA, row_tiler);
    auto gB_row = zipped_divide(mB, row_tiler);

    const uint32_t mA_idx = row_oob ? 0 : mIGather(warp_row_offset);
    const uint32_t mB_idx = row_oob ? 0 : mIScatter(warp_row_offset);

    // 3. copy atom
    using copy_vec = Copy_Atom<Copy_Traits<AutoVectorizingCopyWithAssumedAlignment<128>>, DType>;
    TiledCopy tiled_copy = make_tiled_copy(copy_vec{}, thr_layout, val_layout);
    ThrCopy thr_copy = tiled_copy.get_slice(lane_id);

    auto gA_0 = gA_row(make_coord(_, _), make_coord(0, 0));
    auto gAtA_0 = thr_copy.partition_S(gA_0);
    Tensor rA = make_fragment_like(gAtA_0);

    // 4. mainloop
    auto gA = gA_row(make_coord(_, _), make_coord(mA_idx, bidy));
    auto gB = gB_row(make_coord(_, _), make_coord(mB_idx, bidy));
    auto gAtA = thr_copy.partition_S(gA);
    auto gBtB = thr_copy.partition_S(gB);

    if (!row_oob) {
        copy(copy_vec{}, gAtA, rA);
        #pragma unroll
        for (uint32_t j=0; j < size(rA); j+=atomic_add_width) {
            atomic_add_ptx<DType, atomic_add_width>(&gBtB(j), &rA(j));
        }
    }
}


template <uint32_t kHiddenSize, bool kUsePDL, typename DType>
struct GatherScatterKernel {
    static void run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView IGather,
        const tvm::ffi::TensorView IScatter,
        const tvm::ffi::TensorView B
    ) {
        using namespace host;
        auto T = SymbolicSize{"num_candidate_tokens"};
        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"num_reduced_tokens"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        K.set_value(kHiddenSize);
        device.set_options<kDLCUDA>();

        // host-side checking
        TensorMatcher({T, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(A);
        
        TensorMatcher({M}) //
            .with_strides({1})
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(IGather).verify(IScatter);
        
        TensorMatcher({N, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(B);
        
        static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
        const auto num_candidate_tokens = static_cast<uint32_t>(T.unwrap());
        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        const auto num_reduced_tokens = static_cast<uint32_t>(N.unwrap());

        uint32_t row_per_cta = kNumWarps;
        RuntimeCheck(
            kHiddenSize % (device::kWarpThreads * (16 / sizeof(DType))) == 0,
            "Hidden size is less than minimum required, i.e., 32 * (128b / dtype-width)"
        );

        while ((num_tokens / row_per_cta) < kNumSM * 2 && row_per_cta > 1) row_per_cta >>= 1;

        const auto params = GatherScatterParams{
            .A = A.data_ptr(),
            .IGather = IGather.data_ptr(),
            .IScatter = IScatter.data_ptr(),
            .B = B.data_ptr(),
            .T = num_candidate_tokens,
            .M = num_tokens,
            .N = num_reduced_tokens,
        };

        auto kernel = gather_scatter_kernel<1, kThreadsPerBlock, kHiddenSize, kUsePDL, DType>;
        switch (row_per_cta) {
            case 4:
                kernel = gather_scatter_kernel<4, kThreadsPerBlock / 4, kHiddenSize, kUsePDL, DType>; break;
            case 2:
                kernel = gather_scatter_kernel<2, kThreadsPerBlock / 2, kHiddenSize, kUsePDL, DType>; break;
            case 1:
                kernel = gather_scatter_kernel<1, kThreadsPerBlock, kHiddenSize, kUsePDL, DType>; break;
            default:
                RuntimeCheck(false, "Unsupported row_per_cta");
        }
        
        constexpr uint32_t vec_width = 16 / sizeof(DType);
        const dim3 block_size = {div_ceil(num_tokens, row_per_cta), div_ceil(kHiddenSize, vec_width * (kThreadsPerBlock / row_per_cta)), 1};
        const dim3 thread_size = {kThreadsPerBlock, 1, 1};

        LaunchKernel(block_size, thread_size, device.unwrap())
            .enable_pdl(kUsePDL)(kernel, params);

    }
};

} // namespace