#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

// cusparselt
#include <cuda_runtime_api.h>
#include <cusparseLt.h>

namespace {

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, status);            \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template <typename value_t>
struct cuda_type { };

template <>
struct cuda_type <__half> {
    static constexpr cudaDataType value = CUDA_R_16F;
};

template <>
struct cuda_type <__nv_bfloat16> {
    static constexpr cudaDataType value = CUDA_R_16BF;
};

auto order = CUSPARSE_ORDER_ROW;
auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
auto opB = CUSPARSE_OPERATION_TRANSPOSE;
cudaStream_t stream = nullptr;
float alpha = 1.0f;
float beta = 1.0f;

template <bool kUsePDL, typename DType>
struct StructuredSparseKernel {
    static int init(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView A_desc,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView B_desc,
        const tvm::ffi::TensorView D,
        const tvm::ffi::TensorView D_desc,
        const tvm::ffi::TensorView handle_desc,
        const tvm::ffi::TensorView plan_desc,
        const tvm::ffi::TensorView mm_desc,
        const tvm::ffi::TensorView alg_desc,
        const tvm::ffi::TensorView compress_desc
    ) {
        // initialize
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"intermediate_size"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        auto host_device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        host_device.set_options<kDLCPU>();
        
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
        TensorMatcher({N, M}) //
            .with_strides({M, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(D);
        TensorMatcher({512}) //
            .with_strides({1})
            .with_dtype<uint8_t>()
            .with_device(host_device)
            .verify(handle_desc).verify(plan_desc).verify(mm_desc).verify(alg_desc);
        TensorMatcher({2}) //
            .with_strides({1})
            .with_dtype<int64_t>()
            .with_device(host_device)
            .verify(compress_desc);

        const auto kM = static_cast<int64_t>(M.unwrap());
        const auto kN = static_cast<int64_t>(N.unwrap());
        const auto kK = static_cast<int64_t>(K.unwrap());
        
        // init
        CHECK_CUSPARSE(cusparseLtInit(reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr())));
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(B_desc.data_ptr()),
            kN, kK, kK, 16, cuda_type<DType>::value,
            order, CUSPARSELT_SPARSITY_50_PERCENT
        ));
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(A_desc.data_ptr()),
            kM, kK, kK, 16, cuda_type<DType>::value,
            order
        ));
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            kN, kM, kM, 16, cuda_type<DType>::value,
            order
        ));

        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            opA, opB,
            reinterpret_cast<cusparseLtMatDescriptor_t*>(B_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(A_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            CUSPARSE_COMPUTE_32F
        ));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulAlgSelection_t*>(alg_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            CUSPARSELT_MATMUL_ALG_DEFAULT
        ));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulAlgSelection_t*>(alg_desc.data_ptr())
        ));
        void* B_ptr = B.data_ptr();
        CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
            &B_ptr,
            sizeof(B_ptr)
        ));

        // check the B
        int* is_valid_gpu;
        int is_valid;
        CHECK_CUDA(cudaMalloc((void**) &is_valid_gpu, sizeof(int)));
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            B.data_ptr(),
            is_valid_gpu,
            stream
        ));
        CHECK_CUDA(cudaMemcpyAsync(&is_valid, is_valid_gpu, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (is_valid != 0) {
            std::printf("!!!! The matrix has been pruned in a wrong way. "
                        "cusparseLtMatmul will not provide correct results\n");
            return EXIT_FAILURE;
        }
        CHECK_CUDA(cudaFree(is_valid_gpu));

        // get the compressed + metadata size
        size_t compressed_size, compressed_buffer_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            &compressed_size,
            &compressed_buffer_size
        ));
        reinterpret_cast<int64_t*>(compress_desc.data_ptr())[0] = compressed_size;
        reinterpret_cast<int64_t*>(compress_desc.data_ptr())[1] = compressed_buffer_size;
        return 0;
    }
    
    static int compress(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView dB,
        const tvm::ffi::TensorView handle_desc,
        const tvm::ffi::TensorView plan_desc,
        const tvm::ffi::TensorView compress_desc
    ) {
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"intermediate_size"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        auto host_device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        
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

        const auto kM = static_cast<int64_t>(M.unwrap());
        const auto kN = static_cast<int64_t>(N.unwrap());
        const auto kK = static_cast<int64_t>(K.unwrap());

        size_t compressed_buffer_size = reinterpret_cast<int64_t*>(compress_desc.data_ptr())[1];
        void* compressedBuffer;
        CHECK_CUDA(cudaMalloc((void**) &compressedBuffer, compressed_buffer_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            B.data_ptr(),
            dB.data_ptr(),
            compressedBuffer,
            stream
        ));
        CHECK_CUDA(cudaFree(compressedBuffer));
        return 0;
    }

    static int update(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView A_desc,
        const tvm::ffi::TensorView B_desc,
        const tvm::ffi::TensorView dB,
        const tvm::ffi::TensorView D,
        const tvm::ffi::TensorView D_desc,
        const tvm::ffi::TensorView handle_desc,
        const tvm::ffi::TensorView plan_desc,
        const tvm::ffi::TensorView mm_desc,
        const bool need_tuning
    ) {
        // update A_desc, and re-tuning the kernel
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"intermediate_size"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        auto host_device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        host_device.set_options<kDLCPU>();

        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(A);
        TensorMatcher({N, M}) //
            .with_strides({M, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(D);
        TensorMatcher({512}) //
            .with_strides({1})
            .with_dtype<uint8_t>()
            .with_device(host_device)
            .verify(handle_desc).verify(plan_desc).verify(mm_desc);

        const auto kM = static_cast<int64_t>(M.unwrap());
        const auto kN = static_cast<int64_t>(N.unwrap());
        const auto kK = static_cast<int64_t>(K.unwrap());

        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(A_desc.data_ptr()),
            kM, kK, kK, 16, cuda_type<DType>::value,
            order
        ));
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            kN, kM, kM, 16, cuda_type<DType>::value,
            order
        ));

        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulDescriptor_t*>(mm_desc.data_ptr()),
            opA, opB,
            reinterpret_cast<cusparseLtMatDescriptor_t*>(B_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(A_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatDescriptor_t*>(D_desc.data_ptr()),
            CUSPARSE_COMPUTE_32F
        ));

        if (need_tuning) {
            auto ret = tuning(A, dB, handle_desc, plan_desc, kM, kN, kK);
        }
        return 0;
    }

    static int tuning(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView dB,
        const tvm::ffi::TensorView handle_desc,
        const tvm::ffi::TensorView plan_desc,
        const int64_t kM,
        const int64_t kN,
        const int64_t kK
    ) {
        // tuning
        int num_streams = 0;
        cudaStream_t* streams = nullptr;
        void* C_tmp;
        void* D_tmp;
        CHECK_CUDA(cudaMalloc((void**) &C_tmp, kN * kM * sizeof(DType)));
        CHECK_CUDA(cudaMalloc((void**) &D_tmp, kN * kM * sizeof(DType)));

        CHECK_CUSPARSE(cusparseLtMatmulSearch(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            &alpha, dB.data_ptr(), A.data_ptr(), &beta, C_tmp, D_tmp,
            nullptr,
            streams,
            num_streams
        ));
        CHECK_CUDA(cudaFree(C_tmp));
        CHECK_CUDA(cudaFree(D_tmp));
        return 0;
    }

    static int run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView dB,
        const tvm::ffi::TensorView C,
        const tvm::ffi::TensorView D,
        const tvm::ffi::TensorView handle_desc,
        const tvm::ffi::TensorView plan_desc
    ) {
        using namespace host;
        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"intermediate_size"};
        auto K = SymbolicSize{"hidden_size"};
        auto device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        
        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(A);
        TensorMatcher({N, M}) //
            .with_strides({M, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(C).verify(D);

        const auto kM = static_cast<int64_t>(M.unwrap());
        const auto kN = static_cast<int64_t>(N.unwrap());
        const auto kK = static_cast<int64_t>(K.unwrap());

        // peform spmm
        int num_streams = 0;
        cudaStream_t* streams = nullptr;
        size_t workspace_size;
        void* workspace;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            &workspace_size
        ));
        CHECK_CUDA(cudaMalloc((void**) &workspace, workspace_size));
        CHECK_CUSPARSE(cusparseLtMatmul(
            reinterpret_cast<cusparseLtHandle_t*>(handle_desc.data_ptr()),
            reinterpret_cast<cusparseLtMatmulPlan_t*>(plan_desc.data_ptr()),
            &alpha, dB.data_ptr(), A.data_ptr(), &beta, C.data_ptr(), D.data_ptr(),
            workspace,
            streams,
            num_streams
        ));
        CHECK_CUDA(cudaFree(workspace));
        return 0;
    }
};

} //namespace