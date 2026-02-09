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
    uint64_t M;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

template <typename DType>
__device__ DType swiglu(DType a, DType b) {
    if constexpr (std::is_same_v<DType, fp32_t>) {
        return a * (b * (1.0f / (1.0f + __expf(-b))));
    }
    else if constexpr (std::is_same_v<DType, fp16_t>) {
        float a_f = __half2float(a);
        float b_f = __half2float(b);
        return __float2half(a_f * (b_f * (1.0f / (1.0f + __expf(-b_f)))));
    }
    else if constexpr (std::is_same_v<DType, bf16_t>) {
        float a_f = __bfloat162float(a);
        float b_f = __bfloat162float(b);
        return __float2bfloat16(a_f * (b_f * (1.0f / (1.0f + __expf(-b_f)))));
    }
    else {
        return a * (b * (1.0 / (1.0 + exp(-b))));
    }
};

template <uint32_t kRowPerCTA, uint32_t kColPerCTA, int64_t kHiddenSize, bool kUsePDL, typename DType>
__global__ void swiglu_kernel(const __grid_constant__ SwiGLUParams params) {
    using namespace cute;
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t dimx = gridDim.x;

    const auto& [up, gate, output, M] = params;
    constexpr int64_t K = kHiddenSize;
    constexpr uint32_t vec_width = 16 / sizeof(DType);
    bool is_oob = (dimx * kRowPerCTA != M) && (bidx == dimx - 1);

    // 1. cute make layout
    const auto gA_layout = make_ordered_layout(make_shape(M, Int<K>{}), make_step(_1{}, _0{}));

    Tensor gA = make_tensor(make_gmem_ptr(static_cast<const DType*>(up)), gA_layout);
    Tensor gB = make_tensor(make_gmem_ptr(static_cast<const DType*>(gate)), gA_layout);
    Tensor gC = make_tensor(make_gmem_ptr(static_cast<DType*>(output)), gA_layout);

    // 2. tv layout
    auto thr_layout = make_ordered_layout(make_shape(Int<kRowPerCTA>{}, Int<kColPerCTA>{}), make_step(_1{}, _0{}));
    auto val_layout = make_ordered_layout(make_shape(_1{}, Int<vec_width>{}), make_step(_1{}, _0{}));

    // 3. copy atom
    using copy_pred = Copy_Atom<Copy_Traits<UniversalCopy<uint_bit_t<sizeof(DType) * 8>>>, DType>;
    using copy_vec = Copy_Atom<Copy_Traits<AutoVectorizingCopyWithAssumedAlignment<128>>, DType>;
    
    TiledCopy tiled_copy = make_tiled_copy(copy_pred{}, thr_layout, val_layout);
    ThrCopy thr_copy = tiled_copy.get_slice(tidx);
    auto tiler = product_each(shape(raked_product(thr_layout, val_layout)));
    
    // 4. tiling gmem
    auto gA_tiled = zipped_divide(gA, tiler);
    auto gB_tiled = zipped_divide(gB, tiler);
    auto gC_tiled = zipped_divide(gC, tiler);

    auto gCp = make_identity_tensor(shape(gC));
    auto gCp_tiled = zipped_divide(gCp, tiler);

    auto gAtA_0 = gA_tiled(make_coord(_, _), make_coord(bidx, 0));
    auto gArA_0 = thr_copy.partition_S(gAtA_0);

    Tensor rA = make_fragment_like(gArA_0);
    Tensor rB = make_fragment_like(gArA_0);
    Tensor rC = make_fragment_like(gArA_0);
    Tensor rCp = make_tensor<bool>(shape(rC), stride(rC));

    // 5. partition
    constexpr uint32_t k_range = shape<1,1>(gA_tiled);
    #pragma unroll
    for (uint32_t i=0; i < k_range; ++i) {
        auto gAtA = gA_tiled(make_coord(_, _), make_coord(bidx, i));
        auto gBtB = gB_tiled(make_coord(_, _), make_coord(bidx, i));
        auto gCtC = gC_tiled(make_coord(_, _), make_coord(bidx, i));
        auto gCptCp = gCp_tiled(make_coord(_, _), make_coord(bidx, i));

        auto gArA = thr_copy.partition_S(gAtA);
        auto gBrB = thr_copy.partition_S(gBtB);
        auto gCrC = thr_copy.partition_D(gCtC);
        auto gCprCp = thr_copy.partition_S(gCptCp);

        if (is_oob) {
            #pragma unroll
            for (uint32_t j=0; j < size(gCprCp); ++j) {
                rCp(j) = elem_less(gCprCp(j), shape(gA));
            }
        }

        if (is_oob) {
            copy_if(copy_pred{}, rCp, gArA, rA);
            copy_if(copy_pred{}, rCp, gBrB, rB);
        }
        else {
            copy(copy_vec{}, gArA, rA);
            copy(copy_vec{}, gBrB, rB);
        }
        
        #pragma unroll
        for (uint32_t j=0; j < size(rA); ++j) {
            DType a = rA(j);
            DType b = rB(j);
            rC(j) = swiglu<DType>(a, b);
        }

        if (is_oob) {
            copy_if(copy_pred{}, rCp, rC, gCrC);
        }
        else {
            copy(copy_vec{}, rC, gCrC);
        }

    }

}

template <int64_t kHiddenSize, bool kUsePDL, typename DType>
struct SwiGLUKernel {

    static void run(
        const tvm::ffi::TensorView up,
        const tvm::ffi::TensorView gate,
        const tvm::ffi::TensorView output
    ) {
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        K.set_value(kHiddenSize);
        device.set_options<kDLCUDA>();

        RuntimeCheck(kHiddenSize % 128 == 0, "Hidden size must be divisible by 128");

        // checking on host
        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(up).verify(gate).verify(output);
        
        static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
        uint32_t row_per_cta = 4;
        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        while ((num_tokens / row_per_cta) < kNumSM * 2 && row_per_cta > 1) row_per_cta >>= 1;
        
        const auto params = SwiGLUParams{
            .up = up.data_ptr(),
            .gate = gate.data_ptr(),
            .output = output.data_ptr(),
            .M = num_tokens,
        };
        
        auto kernel = swiglu_kernel<1, kThreadsPerBlock, kHiddenSize, kUsePDL, DType>;
        switch (row_per_cta) {
            case 4:
                kernel = swiglu_kernel<4, kThreadsPerBlock / 4, kHiddenSize, kUsePDL, DType>; break;
            case 2:
                kernel = swiglu_kernel<2, kThreadsPerBlock / 2, kHiddenSize, kUsePDL, DType>; break;
            case 1:
                kernel = swiglu_kernel<1, kThreadsPerBlock, kHiddenSize, kUsePDL, DType>; break;
            default:
                throw std::runtime_error("row_per_cta must be 1, 2, 4, 8");
        }

        LaunchKernel(div_ceil(num_tokens, row_per_cta), kThreadsPerBlock, device.unwrap())
            .enable_pdl(kUsePDL)(kernel, params);
    }
};


} // namespace
