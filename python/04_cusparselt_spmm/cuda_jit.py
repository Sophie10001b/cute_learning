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
    DEFAULT_LDFLAGS,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

@cache_once
def _jit_cusparselt_spmm_module(
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        "cusparselt_spmm",
        *args,
        cuda_files=[str(ROOT_PATH / "include" / "sparse_tensor_core_cusparselt.cuh")],
        cuda_wrappers=[
            ("init", f"StructuredSparseKernel<{args}>::init"),
            ("compress", f"StructuredSparseKernel<{args}>::compress"),
            ("update", f"StructuredSparseKernel<{args}>::update"),
            ("run", f"StructuredSparseKernel<{args}>::run"),
        ],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
        extra_ldflags=DEFAULT_LDFLAGS,
    )


class cuSPARSELtLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.nn.Parameter,
        mask: torch.Tensor,
        cache_num: int=10,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight * mask, requires_grad=False)
        self.N, self.K = self.weight.shape
        self._cache_num = cache_num
        self._plan_cache = {}

        dtype = self.weight.dtype
        device = self.weight.device

        # init cuSPARSELt workspace
        self.A_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.B_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.D_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.handle_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.plan_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.mm_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')
        self.alg_desc = torch.empty((512,), dtype=torch.uint8, device='cpu')

        # get dummy inputs
        dummy_input = torch.empty((512, self.K), dtype=dtype, device=device)
        dummy_output = torch.empty((self.N, 512), dtype=dtype, device=device)

        # init
        compress_desc = torch.zeros((2,), dtype=torch.int64, device='cpu')
        self.cusparselt_wrapper = _jit_cusparselt_spmm_module(dtype)
        self.cusparselt_wrapper.init(
            dummy_input, self.A_desc, self.weight, self.B_desc, dummy_output, self.D_desc,
            self.handle_desc, self.plan_desc, self.mm_desc, self.alg_desc, compress_desc,
        )

        # compress
        weight_tmp = torch.empty((compress_desc[0],), dtype=torch.int8, device=device)
        self.cusparselt_wrapper.compress(
            dummy_input, self.weight, weight_tmp, self.handle_desc, self.plan_desc, compress_desc,
        )
        self.weight = torch.nn.Parameter(weight_tmp, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. check cache for M, if not found, create a new plan and auto-tuning
        M = x.shape[0]
        C = torch.zeros((self.N, x.shape[0]), dtype=x.dtype, device=x.device)
        D = torch.empty_like(C)

        if M not in self._plan_cache:
            self.cusparselt_wrapper.update(
                x, self.A_desc, self.B_desc, self.weight, D, self.D_desc,
                self.handle_desc, self.plan_desc, self.mm_desc, True
            )
            if len(self._plan_cache) >= self._cache_num:
                self._plan_cache.pop(next(iter(self._plan_cache)))
            
            self._plan_cache[M] = deepcopy(self.plan_desc)

        self.cusparselt_wrapper.run(
            x, self.weight, C, D, self.handle_desc, self._plan_cache[M],
        )
        return D.t()

def random_sample(
    shape,
    sparsity: Optional[float]=0.5,
    block_size: Optional[int]=4,
    device: Optional[torch.device]=None,
    **kwargs,
) -> torch.Tensor:
    left = int(sparsity * block_size)
    right = block_size
    assert left > 0

    mask = torch.ones(shape, device=device).flatten()
    mask = rearrange(mask, '(a b) -> a b', b=right)
    indices = torch.multinomial(mask, left, replacement=False)

    rows = torch.arange(mask.shape[0], device=device).view(-1, 1).expand(-1, left)
    mask[rows, indices] = False
    mask = mask.reshape(shape)
    return mask.to(torch.bool)

if __name__ == "__main__":
    dtype = torch.float16
    device = 'cuda:0'

    import random
    from copy import deepcopy
    from torch.sparse.semi_structured import SparseSemiStructuredTensorCUSPARSELT
    
    weight = torch.rand((4096, 4096), dtype=dtype, device=device)
    mask = random_sample(weight.shape, device=device)
    linear = cuSPARSELtLinear(deepcopy(weight), mask)

    linear_ref = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=dtype, device=device)
    linear_ref.weight = torch.nn.Parameter(deepcopy(weight), requires_grad=False)
    linear_ref.weight *= mask
    linear_ref.weight = torch.nn.Parameter(SparseSemiStructuredTensorCUSPARSELT.from_dense(linear_ref.weight), requires_grad=False)

    for i in range(5):
        M = random.randint(1, 1024)
        # M = 512
        x = torch.rand((M, weight.shape[1]), dtype=dtype, device=device)

        jit_res = linear(x)
        ref_res = linear_ref(x)
        assert torch.allclose(jit_res, ref_res, rtol=1e-2, atol=1e-2)