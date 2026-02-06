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

if TYPE_CHECKING:
    from tvm_ffi.module import Module

@cache_once
def _jit_rmsnorm_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "rmsnorm",
        *args,
        cuda_files=["/root/autodl-tmp/cute_learning/include/swiglu.cuh"],
        cuda_wrappers=[("rmsnorm", f"RMSNormKernel<{args}>::run")],
        extra_include_paths=["/root/autodl-tmp/cute_learning/3rdparty/cutlass/include"]
    )


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_rmsnorm_module(input.shape[-1], input.dtype)
    module.rmsnorm(input, weight, output, eps)

if __name__ == '__main__':
    a = torch.randn(512, 1024, dtype=torch.float16, device='cuda')
    b = torch.randn(1024, dtype=torch.float16, device='cuda')
    c = torch.empty_like(a)

    rmsnorm(a, b, c)
    pass
