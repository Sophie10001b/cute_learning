from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

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
    "Identity": 0,
    "ReLU": 1,
    "SiLU": 2,
}

@cache_once
def _jit_gather_scatter_gemm_module(
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    num_group: int,
    dtype: torch.dtype,
    activation: str,
) -> Module:
    args = make_cpp_args(
        hidden_size, intermediate_size, group_size, num_group, is_arch_support_pdl(), dtype, ACTIVATION.get(activation, 0)
    )
    return load_jit(
        "gather_scatter_gemm_sm80",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "gather_scatter_gemm_sm80.cuh")],
        cuda_wrappers=[("gather_scatter_gemm_sm80", f"GatherScatterGEMMKernelSM80<{args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )


def gather_scatter_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    C: Optional[torch.Tensor]=None,
    activation: Optional[str]="Identity",
    estimate_sparsity: Optional[float]=0.5,
) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 2
    bsz, seqlen, K = A.shape
    A_F = A.flatten(0, 1)

    M, K = A_F.shape
    N, _ = B.shape
    if C is None:
        C = torch.zeros((M, N), dtype=A.dtype, device=A.device)
    
    if Mask.dim() == 2: Mask = Mask.unsqueeze(0)
    Mask_T = Mask.flatten(0, 1).transpose(0, 1).contiguous()
    Mask_ST, Index = torch.sort(Mask_T, dim=-1, descending=True, stable=False)

    NG = Mask.shape[-1]
    G = K // NG

    module = _jit_gather_scatter_gemm_module(K, N, G, NG, A.dtype, activation)
    module.gather_scatter_gemm_sm80(A_F, B, Mask_ST.to(torch.uint8), Index.to(torch.uint32), C, estimate_sparsity)
    return C.reshape(bsz, seqlen, N)

def test(bsz: int, seqlen: int, N: int, K: int, G: int, sparsity: float):
    A = torch.randn((bsz, seqlen, K), dtype=torch.float16, device="cuda")
    B = torch.randn((N, K), dtype=torch.float16, device="cuda")
    Mask = torch.rand((bsz, seqlen, N // G), device="cuda") > sparsity
    C = gather_scatter_gemm(A, B, Mask, activation="SiLU", estimate_sparsity=sparsity)

if __name__ == "__main__":
    test(1, 16, 4096, 4096, 128, 0.25)