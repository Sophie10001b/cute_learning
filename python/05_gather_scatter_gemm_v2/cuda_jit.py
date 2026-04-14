from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from einops import rearrange
from copy import deepcopy

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from utils import (
    ROOT_PATH,
    THIRD_PARTY_HEADER_DIRS,
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

ACTIVATION = {
    "identity": 0,
    "relu": 1,
    "silu": 2,
}

SMEM_SIZE = {
    80: 167963,
    90: 233472,
    100: 233472,
}

def next_pow_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y

@cache_once
def _jit_gather_scatter_gemm_module(
    N: int, K: int, NG: int, NGIter: int,
    BM: Optional[int]=64,
    BN: Optional[int]=64,
    BK: Optional[int]=64,
    SplitK: Optional[int]=1,
    Pipeline: Optional[int]=3,
    dtype: Optional[torch.dtype]=torch.float16,
    activation: Optional[str]="identity",
) -> Module:
    args = make_cpp_args(
        N, K, NG, NGIter,
        BM, BN, BK, SplitK, Pipeline,
        is_arch_support_pdl(), dtype, ACTIVATION.get(activation, 0)
    )
    return load_jit(
        "gather_scatter_gemm_sm80_sm120",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "gather_scatter_gemm_sm80_sm120.cuh")],
        cuda_wrappers=[("gather_scatter_gemm_sm80_sm120", f"GatherScatterGEMMKernel<{args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )

def gather_scatter_gemm_cute(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor]=None,
    activation: Optional[str]="identity",
    estimate_sparsity: Optional[float]=0.5,
) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 2
    assert Mask.dim() in (2, 3)
    assert estimate_sparsity <= 1

    Bsz, T, K = A.shape
    A_f = A.flatten(0, 1)

    M, _ = A_f.shape
    N, _ = B.shape
    if D is None: D = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    if Mask.dim() == 2: Mask = Mask.unsqueeze(-1) # [B, T, NG]
    Mask_t = Mask.flatten(0, 1).transpose(0, 1).contiguous()
    Mask_st, Index = torch.sort(Mask_t, dim=-1, descending=True, stable=False)

    NG = Mask_st.shape[0]

    # heuristic tiling
    G = N // NG
    cc_major, cc_minor = torch.cuda.get_device_capability()
    smem = SMEM_SIZE.get(cc_major * 10 + cc_minor, 101376)
    num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
    data_width = A.dtype.itemsize

    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity
    BM = next_pow_of_2(int(M * estimate_sparsity))
    if BM >= int(M * estimate_sparsity): BM >>= 1

    if cc_major in (8, 12): # sm80-like warp mma
        BM = min(128, max(16, BM))
        BK = 64
        BN = min(G, 256)
        Pipeline = 3

        while data_width * Pipeline * (BM * BK + BN * BK) > smem / 2 and BK > 32: # sm80, sm120 need 2+ CTA per SMs
            BK >>= 1
        while data_width * Pipeline * (BM * BK + BN * BK) > smem / 2 and BN > 32:
            BN >>= 1
        
        SplitK = 1
        base_tile_num = cdiv(M, BM) * cdiv(N, BN)
        if base_tile_num < num_sm:
            min_waste = 1.0
            best_split_k = 2
            for split_k in [2, 4, 8]:
                waste = float(num_sm - ((base_tile_num * split_k) % num_sm)) / float(num_sm)
                if (min_waste > 0 and waste < min_waste):
                    min_waste = waste
                    best_split_k = split_k
            
            SplitK = best_split_k
    
    elif cc_major in (9,): # sm90-like wgmma
        raise NotImplementedError("sm90-like wgmma is not supported")
    
    elif cc_major in (10,): # sm100-like umma
        raise NotImplementedError("sm100-like umma is not supported")
    
    else:
        raise ValueError(f"Unsupported compute capability {cc_major}.{cc_minor}")
    
    module = _jit_gather_scatter_gemm_module(
        N, K, NG, G // BN, BM, BN, BK, SplitK, Pipeline,
        A.dtype, activation,
    )
    module.gather_scatter_gemm_sm80_sm120(
        A_f, B, Mask_st.to(torch.uint8), Index.to(torch.uint32), D, estimate_sparsity,
    )
    return D.reshape(Bsz, T, N)

def ref_program(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor]=None,
    activation: Optional[str]="identity",
    estimate_sparsity: Optional[float]=0.5,
) -> torch.Tensor:
    D = A @ B.T
    D = rearrange(D, 'b t (ng d) -> b t ng d', ng=Mask.shape[-1])
    if Mask.dim() == 2: Mask = Mask.unsqueeze(-1) # [B, T, NG]

    D.masked_fill_(Mask.logical_not()[:, :, :, None], 0)

    if activation == 'silu': D = D * torch.sigmoid(D)
    elif activation == 'relu': D = torch.relu(D)

    return rearrange(D, 'b t ng d -> b t (ng d)')


if __name__ == "__main__":
    device = 'cuda:0'
    dtype = torch.float16

    M = 4096
    N = 4096
    K = 4096
    NG = 32

    A = torch.rand((1, M, K), dtype=dtype, device=device)
    B = torch.rand((N, K), dtype=dtype, device=device)
    Mask = torch.rand((1, M, NG), device=device) > 0.5
    D = torch.zeros((M, N), dtype=dtype, device=device)

    D_cute = gather_scatter_gemm_cute(
        deepcopy(A), deepcopy(B), deepcopy(Mask), deepcopy(D), activation='identity', estimate_sparsity=0.5
    )