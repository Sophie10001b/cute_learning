import pathlib
from functools import lru_cache

@lru_cache()
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
    return cur_dir

ROOT_PATH = _resolve_kernel_path()
THIRD_PARTY_HEADER_DIR = _resolve_kernel_path() / "3rdparty"
THIRD_PARTY_HEADER_DIRS = [
    str(THIRD_PARTY_HEADER_DIR / "cutlass/include"),
    str(THIRD_PARTY_HEADER_DIR / "dlpack/include"),
    str(THIRD_PARTY_HEADER_DIR / "tvm-ffi/include"),
    str(THIRD_PARTY_HEADER_DIR / "sglang/include"),
]

DEFAULT_CFLAGS = []
DEFAULT_CUDA_CFLAGS = []
