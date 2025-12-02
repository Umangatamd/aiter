# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/om/comkstsgay7hfrdfg54yrvp34kgx6xjukkherqnqb7qosffe5ttk.py
# Topologically Sorted Source Nodes: [da, sig, mul_2, sub, mul_3, add, dh0, dh1], Original ATen: [aten.mul, aten.sigmoid, aten.rsub, aten.add]
# Source node to ATen node mapping:
#   add => add
#   da => mul
#   dh0 => mul_4
#   dh1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   sig => sigmoid
#   sub => sub
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %arg3_1), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%arg5_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sigmoid), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, %sub), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1.0), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, %arg4_1), kwargs = {})
triton_poi_fused_add_mul_rsub_sigmoid_0 = async_compile.triton('triton_poi_fused_add_mul_rsub_sigmoid_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_rsub_sigmoid_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '3A611C8265A34DABC76FB04FDE957BE9FB324CF500917CC372D2F75D272D299B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_rsub_sigmoid_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp11 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tmp6 = 1.0
    tmp7 = tmp6 - tmp4
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8 + tmp6
    tmp10 = tmp5 * tmp9
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 3072), (3072, 1))
    assert_size_stride(arg1_1, (8, 3072), (3072, 1))
    assert_size_stride(arg2_1, (8, 8192), (8192, 1))
    assert_size_stride(arg3_1, (8, 8192), (8192, 1))
    assert_size_stride(arg4_1, (8, 8192), (8192, 1))
    assert_size_stride(arg5_1, (8, 8192), (8192, 1))
    assert_size_stride(arg6_1, (8192, 3072), (3072, 1))
    assert_size_stride(arg7_1, (8192, 3072), (3072, 1))
    assert_size_stride(arg8_1, (8, 3072), (3072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 8192), (8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dg], Original ATen: [aten.mm]
        extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3072, 8192), (1, 3072), 0), out=buf0)
        del arg0_1
        buf1 = empty_strided_cuda((8, 8192), (8192, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 8192), (8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [da, sig, mul_2, sub, mul_3, add, dh0, dh1], Original ATen: [aten.mul, aten.sigmoid, aten.rsub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_rsub_sigmoid_0.run(buf0, arg3_1, arg5_1, arg4_1, buf1, buf3, 65536, stream=stream0)
        del arg3_1
        del arg4_1
        del arg5_1
        del buf0
        buf2 = empty_strided_cuda((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, arg6_1, out=buf2)
        del arg6_1
        buf4 = empty_strided_cuda((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dh1, addmm], Original ATen: [aten.mul, aten.addmm]
        extern_kernels.addmm(buf2, buf3, arg7_1, alpha=1, beta=1, out=buf4)
        del arg7_1
        del buf2
        buf5 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dw_gate], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (8192, 8), (1, 8192), 0), arg8_1, out=buf5)
        del buf1
        buf6 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dw_value], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (8192, 8), (1, 8192), 0), arg8_1, out=buf6)
        del arg8_1
        del buf3
        buf7 = empty_strided_cuda((8192, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dw_down], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg2_1, (8192, 8), (1, 8192), 0), arg1_1, out=buf7)
        del arg1_1
        del arg2_1
    return (buf4, buf5, buf6, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
