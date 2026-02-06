import os
import torch
import triton

from functools import partial, lru_cache
from typing import Optional, List, Dict, Callable
from triton.testing import do_bench, do_bench_cudagraph

import cutlass
import cutlass.cute as cute

TYPE_CONVERT = {
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

### triton ref
import triton
import triton.language as tl

@triton.jit
def triton_device(
    a: tl.tensor,
    b: tl.tensor,
    c: tl.tensor,
    op: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    tid = tl.program_id(0)

    k_offset = tl.arange(0, BK)
    for i in tl.range(tl.cdiv(K, BK)):
        a_data = tl.load(
            a + tid * K + k_offset,
            mask=k_offset < K,
            other=0,
        )
        b_data = tl.load(
            b + tid * K + k_offset,
            mask=k_offset < K,
            other=0,
        )

        a_data = a_data.to(tl.float32)
        b_data = b_data.to(tl.float32)

        if op == 'add':
            c_data = a_data + b_data
        
        if op == 'silu':
            c_data = b_data * (a_data * tl.sigmoid(a_data))

        tl.store(
            c + tid * K + k_offset,
            c_data.to(c.dtype.element_ty),
            mask=k_offset < K,
        )

        k_offset += BK

def triton_host(
    a: torch.Tensor,
    b: torch.Tensor,
    op: str,
):
    assert a.shape == b.shape
    BK = 128
    c = torch.empty_like(a)
    grid = lambda META: (a.shape[0],)
    triton_device[grid](a, b, c, op, a.shape[1], BK)
    return c


@cute.kernel
def cute_device(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_pred: cute.TiledCopy,
    row_aligned: cutlass.Boolean,
    epilogue_op: cutlass.Constexpr=lambda x, y:x,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    k_tile_range = mC.shape[1][1]

    mPred = cute.make_identity_tensor(mC.shape)
    need_pred = (bidx == mC.shape[1][0] - 1) & (row_aligned)

    thr_copy_A = tiled_copy_pred.get_slice(tidx)
    thr_copy_B = tiled_copy_pred.get_slice(tidx)
    thr_copy_C = tiled_copy_pred.get_slice(tidx)

    for k_tile in cutlass.range_constexpr(k_tile_range):
        gA = mA[(None, None), (bidx, k_tile)]
        gB = mB[(None, None), (bidx, k_tile)]
        gC = mC[(None, None), (bidx, k_tile)]
        gPred = mPred[(None, None), (bidx, k_tile)]

        # print(f'[DSL INFO] gA shape: {gA.type}')

        tA = thr_copy_A.partition_S(gA)
        tB = thr_copy_B.partition_S(gB)
        tC = thr_copy_C.partition_S(gC)
        tPred = thr_copy_C.partition_S(gPred)

        # print(f'[DSL INFO] tA shape: {tA.type}')
        # print(f'[DSL INFO] tPred shape: {tPred.type}')

        fragA = cute.make_fragment_like(tA)
        fragB = cute.make_fragment_like(tB)
        fragC = cute.make_fragment_like(tC)
        fragPred = cute.make_fragment_like(tPred, dtype=cutlass.Boolean)

        for i in range(cute.size(fragPred)):
            val = cute.elem_less(tPred[i], mC.shape)
            fragPred[i] = val

        if need_pred:
            cute.copy(tiled_copy_pred, tA, fragA, pred=fragPred)
            cute.copy(tiled_copy_pred, tB, fragB, pred=fragPred)
        else:
            cute.autovec_copy(tA, fragA)
            cute.autovec_copy(tB, fragB)

        res = epilogue_op(fragA.load().to(cutlass.Float32), fragB.load().to(cutlass.Float32))
        res = res.to(mC.element_type)
        fragC.store(res)

        if need_pred:
            cute.copy(tiled_copy_pred, fragC, tC, pred=fragPred)
        else:
            cute.autovec_copy(fragC, tC)


@cute.jit
def cute_host(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    op: cutlass.Constexpr,
    stream,
):
    threads = 256
    dtype = a.element_type
    width = dtype.width

    thr_layout = cute.make_ordered_layout((4, threads // 4), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, (128 // width) * 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    tiled_copy_pred = cute.make_tiled_copy(
        atom=cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
        ),
        layout_tv=tv_layout,
        tiler_mn=tiler_mn,
    )

    mA = cute.zipped_divide(a, tiler_mn)
    mB = cute.zipped_divide(b, tiler_mn)
    mC = cute.zipped_divide(c, tiler_mn)

    row_aligned = a.shape[0] == (mA.shape[0][0] * mA.shape[1][0])

    cute_device(mA, mB, mC, tiled_copy_pred, row_aligned, op).launch(
        grid=[mC.shape[1][0], 1, 1],
        block=[cute.size(thr_layout), 1, 1],
        stream=stream,
    )

@lru_cache
def cute_compile(k: int, dtype: torch.dtype, op: Callable):
    m = cute.sym_int()
    a_cute = cute.runtime.make_fake_compact_tensor(dtype=TYPE_CONVERT[dtype], shape=(m, k), stride_order=(1, 0), assumed_align=16)
    b_cute = cute.runtime.make_fake_compact_tensor(dtype=TYPE_CONVERT[dtype], shape=(m, k), stride_order=(1, 0), assumed_align=16)
    c_cute = cute.runtime.make_fake_compact_tensor(dtype=TYPE_CONVERT[dtype], shape=(m, k), stride_order=(1, 0), assumed_align=16)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(cute_host, a_cute, b_cute, c_cute, op, stream, options="--enable-tvm-ffi")


@cutlass.dsl_user_op
def add(a, b, *, loc=None, ip=None):
    return a + b

@cutlass.dsl_user_op
def silu(a, b, *, loc=None, ip=None):
    return b * (a * (1.0 / (1.0 + cute.exp(-a, fastmath=True))))

OP_MAP = {
    "add": add,
    "silu": silu,
}

OP_REF_MAP = {
    "add": lambda a, b: a + b,
    "silu": lambda a, b: torch.nn.functional.silu(a) * b
}

def cute_ref(a: torch.Tensor, b: torch.Tensor, op: Callable) -> torch.Tensor:
    c = torch.empty_like(a)

    fn = cute_compile(a.shape[1], a.dtype, op)
    fn(a, b, c)
    return c

def triton_ref(a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
    return triton_host(a, b, op)

@torch.compile
def torch_compile_ref(a: torch.Tensor, b: torch.Tensor, op: Callable) -> torch.Tensor:
    assert a.shape == b.shape
    return op(a, b)

def torch_ref(a: torch.Tensor, b: torch.Tensor, op: Callable) -> torch.Tensor:
    assert a.shape == b.shape
    return op(a, b)

@triton.testing.perf_report((
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[2 ** i for i in range(9, 16)],
        x_log=True,
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'triton', 'cutedsl'],  # Possible values for `line_arg`.
        line_names=['torch', 'triton', 'cutedsl'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='element_wise_op',  # Name for the plot. Used also as a file name for saving the plot.
        args={
            'K': 8192,
            'dtype': torch.bfloat16,
            'op': 'silu',
            'mode': '',
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
))
def benchmark(M: int, K: int, dtype: torch.dtype, op: str, mode: str, provider: str):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(M, K, device="cuda", dtype=dtype)
    op_func = OP_MAP[op]
    op_ref_func = OP_REF_MAP[op]
    if provider == 'torch':
        func = partial(torch_ref, a=a, b=b, op=op_ref_func)
    elif provider == 'triton':
        func = partial(triton_ref, a=a, b=b, op=op)
    elif provider == 'cutedsl':
        func = partial(cute_ref, a=a, b=b, op=op_func)
    
    quantiles = [0.5, 0.2, 0.8]
    with torch.no_grad():
        try:
            if 'cudagraph' in mode:
                ms, min_ms, max_ms = do_bench_cudagraph(func, rep=500, quantiles=quantiles)
            else:
                ms, min_ms, max_ms = do_bench(func, warmup=100, rep=500, quantiles=quantiles)
            print(f"✅ finish {provider} in [M:{M}, K:{K}], mode: {mode}")
        except Exception as e:
            print(e)
            return 0, 0, 0
    
    width = int(str(dtype)[-2:]) / 8

    gbps = lambda ms: (2 * M * K * width * 1e-9) / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)

def main(m_range: List[int], k: int, op: str):
    op_func = partial(OP_MAP[op])
    op_ref_func = partial(OP_REF_MAP[op])
    for m in m_range:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(m, k, device="cuda")
        c_ref = torch_ref(a, b, op_ref_func)
        c_cute = cute_ref(a, b, op_func)
        c_triton = triton_ref(a, b, op)
        assert torch.allclose(c_ref, c_cute, atol=1e-2, rtol=1e-2)
        assert torch.allclose(c_ref, c_triton, atol=1e-2, rtol=1e-2)

        print(f"✅ pass {op} in [M:{m}, K:{k}], max diff: {torch.max(torch.abs(c_ref - c_cute))}")
    
    benchmark.run(print_data=True)

def trace(m_range: List[int], k: int, op: str):
    op_func = partial(OP_MAP[op])
    op_ref_func = partial(OP_REF_MAP[op])
    m = m_range[-1]
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(m, k, device="cuda")

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
        c_ref = torch_ref(a, b, op_ref_func)
        c_cute = cute_ref(a, b, op_func)
        c_triton = triton_ref(a, b, op)
        profiler.step()

    profiler.stop()

if __name__ == "__main__":
    m_range = [2 ** i for i in range(9, 16)]
    k = 8192
    op = 'silu'

    main(m_range, k, op)