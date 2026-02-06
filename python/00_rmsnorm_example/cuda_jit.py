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

@cache_once
def _jit_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "example.cuh")],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_rmsnorm_module(input.shape[-1], input.dtype)
    module.rmsnorm(input, weight, output, eps)

def test():
    a = torch.randn(512, 1024, dtype=torch.float16, device='cuda')
    b = torch.randn(1024, dtype=torch.float16, device='cuda')
    c = torch.empty_like(a)
    eps = 1e-6

    ref = torch.nn.functional.rms_norm(a, [a.shape[-1]], b, eps)
    rmsnorm(a, b, c, eps)

    assert torch.allclose(c, ref, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test()