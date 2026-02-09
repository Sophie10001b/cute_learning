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
def _jit_swiglu_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "swiglu",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "swiglu.cuh")],
        cuda_wrappers=[("swiglu", f"SwiGLUKernel<{args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )


def swiglu(
    up: torch.Tensor,
    gate: torch.Tensor,
    output: Optional[torch.Tensor]=None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(up)
    
    module = _jit_swiglu_module(up.shape[-1], up.dtype)
    module.swiglu(up, gate, output)
    return output

def swiglu_torch(
    up: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return up * (gate * torch.sigmoid(gate))

@torch.compile
def swiglu_torch_compile(
    up: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return up * (gate * torch.sigmoid(gate))


def test(m: int, k: int):
    up = torch.randn(m, k, dtype=torch.float16, device='cuda')
    gate = torch.randn(m, k, dtype=torch.float16, device='cuda')

    ref = up * (gate * torch.sigmoid(gate))
    output = swiglu(up, gate)

    assert torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    print(f"✅ pass swiglu in [M:{m}, K:{k}], max diff: {torch.max(torch.abs(ref - output))}")

def trace(m: int, k: int):
    up = torch.randn(m, k, device="cuda")
    gate = torch.randn(m, k, device="cuda")

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs', use_gzip=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    profiler.start()

    for i in range(10):
        ref1 = swiglu_torch(up, gate)
        ref2 = swiglu_torch_compile(up, gate)
        output = swiglu(up, gate)
        profiler.step()

    profiler.stop()


if __name__ == "__main__":
    trace(8192, 8192)
    # test(8111, 8192)