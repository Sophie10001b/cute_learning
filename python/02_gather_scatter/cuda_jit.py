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
def _jit_gather_scatter_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "gather_scatter",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "gather_scatter.cuh")],
        cuda_wrappers=[("gather_scatter", f"GatherScatterKernel<{args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )


def gather_scatter(
    src: torch.Tensor,
    gather_index: torch.Tensor,
    scatter_index: torch.Tensor,
    dim_size: int,
    dst: Optional[torch.Tensor]=None,
) -> torch.Tensor:
    if dst is None:
        dst = torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    
    assert dst.shape[0] == dim_size
    
    module = _jit_gather_scatter_module(src.shape[-1], src.dtype)
    module.gather_scatter(src, gather_index, scatter_index, dst)
    return dst

def gather_scatter_torch(
    src: torch.Tensor,
    gather_index: torch.Tensor,
    scatter_index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    tmp = torch.gather(src, 0, gather_index.unsqueeze(-1).expand(-1, src.shape[-1]))
    dst = torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    dst.scatter_add_(0, scatter_index.unsqueeze(-1).expand(-1, src.shape[-1]), tmp)
    return dst

@torch.compile
def gather_scatter_torch_compile(
    src: torch.Tensor,
    gather_index: torch.Tensor,
    scatter_index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    tmp = torch.gather(src, 0, gather_index.unsqueeze(-1).expand(-1, src.shape[-1]))
    dst = torch.zeros((dim_size, src.shape[-1]), dtype=src.dtype, device=src.device)
    dst.scatter_add_(0, scatter_index.unsqueeze(-1).expand(-1, src.shape[-1]), tmp)
    return dst


def test(t: int, m: int, n: int, k: int, dtype: torch.dtype):
    A = torch.rand((t, k), dtype=dtype, device="cuda")
    gather_index = torch.randint(0, t, (m,), dtype=torch.int32, device="cuda")
    scatter_index = torch.randint(0, n, (m,), dtype=torch.int32, device="cuda")

    ref = gather_scatter_torch(A, gather_index, scatter_index, n)
    out = gather_scatter(A, gather_index, scatter_index, n)

    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"✅ pass gather_scatter in [T:{t}, M:{m}, N:{n}, K:{k}], max diff: {torch.max(torch.abs(ref - out))}")

def trace(t: int, m: int, n: int, k: int, dtype: torch.dtype):
    A = torch.rand((t, k), dtype=dtype, device="cuda")
    gather_index = torch.randint(0, t, (m,), dtype=torch.int32, device="cuda")
    scatter_index = torch.randint(0, n, (m,), dtype=torch.int32, device="cuda")

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
        ref1 = gather_scatter_torch(A, gather_index, scatter_index, n)
        ref2 = gather_scatter_torch_compile(A, gather_index, scatter_index, n)
        out = gather_scatter(A, gather_index, scatter_index, n)
        profiler.step()

    profiler.stop()


if __name__ == "__main__":
    # test(16384, 8192, 4096, 4096, dtype=torch.float16)
    trace(16384, 8192, 4096, 4096, dtype=torch.float16)