# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Triton backward kernel for ff_a16w16_fused_gated - OpenEvolve format.

Ported from torch.compile generated code.

The backward pass consists of:
- 1 Triton kernel for fused element-wise ops (activation backward + gating backward)
- External GEMM calls for matrix multiplications (uses torch.mm)

Forward: Y = (SiLU(X @ W_gate^T) * (X @ W_value^T)) @ W_down
"""

######################################## Imports ######################################## 
import torch
import triton
import triton.language as tl
from typing import Tuple

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}
######################################## Imports ########################################


# =============================================================================
# TRITON KERNEL: Fused element-wise backward (from torch.compile)
# =============================================================================
# Original: triton_poi_fused_add_mul_rsub_sigmoid_0
# Computes: dh0 = da * sig * (1 + h0 * (1 - sig)), dh1 = dg * a
# Where: da = dg * h1, sig = sigmoid(h0)

@triton.jit
def triton_fused_gating_activation_backward(
    dg_ptr,      # in_ptr0: dg (M, N_half)
    h1_ptr,      # in_ptr1: h1 (M, N_half)
    h0_ptr,      # in_ptr2: h0 (M, N_half)
    a_ptr,       # in_ptr3: a (M, N_half)
    dh0_ptr,     # out_ptr0: dh0 (M, N_half)
    dh1_ptr,     # out_ptr1: dh1 (M, N_half)
    xnumel,
    XBLOCK: tl.constexpr,
):
    """
    Fused backward through gating and activation:
        da = dg * h1
        sig = sigmoid(h0)
        dh0 = da * sig * (1 + h0 * (1 - sig))  # SiLU backward
        dh1 = dg * a
    """
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    
    x0 = xindex
    
    # Load inputs
    dg = tl.load(dg_ptr + x0, mask=xmask)
    h1 = tl.load(h1_ptr + x0, mask=xmask)
    h0 = tl.load(h0_ptr + x0, mask=xmask)
    a = tl.load(a_ptr + x0, mask=xmask)
    
    # Compute da = dg * h1
    da = dg * h1
    
    # Compute dh0 = da * SiLU'(h0)
    # SiLU'(x) = sig(x) * (1 + x * (1 - sig(x)))
    sig = tl.sigmoid(h0)
    dh0 = da * sig * (1.0 + h0 * (1.0 - sig))
    
    # Compute dh1 = dg * a
    dh1 = dg * a
    
    # Store outputs
    tl.store(dh0_ptr + x0, dh0, mask=xmask)
    tl.store(dh1_ptr + x0, dh1, mask=xmask)


# =============================================================================
# PYTHON WRAPPER (mirrors torch.compile's call function)
# =============================================================================

def ff_a16w16_fused_gated_backward(
    dy: torch.Tensor,           # (M, K) - upstream gradient
    x: torch.Tensor,            # (M, K) - input from forward
    w_gate: torch.Tensor,       # (N_half, K) - gate weights
    w_value: torch.Tensor,      # (N_half, K) - value weights  
    w_down: torch.Tensor,       # (N_half, K) - down projection weights
    h0: torch.Tensor,           # (M, N_half) - pre-activation gate
    h1: torch.Tensor,           # (M, N_half) - value branch
    a: torch.Tensor,            # (M, N_half) - post-activation
    g: torch.Tensor,            # (M, N_half) - gated output
    XBLOCK: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton backward pass for fused gated feed-forward.
    
    Ported from torch.compile generated code:
    - Uses torch.mm for GEMMs (like extern_kernels.mm)
    - Uses Triton kernel for fused element-wise ops
    
    Returns: dx, dw_gate, dw_value, dw_down
    """
    M, K = dy.shape
    N_half = w_gate.shape[0]
    
    device = dy.device
    dtype = dy.dtype
    
    # Step 1: dg = dy @ W_down^T  (extern_kernels.mm in torch.compile)
    dg = torch.mm(dy, w_down.T)  # (M, K) @ (K, N_half) = (M, N_half)
    
    # Step 2: Fused element-wise backward (Triton kernel)
    # Computes dh0 and dh1 from dg, h1, h0, a
    dh0 = torch.empty(M, N_half, device=device, dtype=dtype)
    dh1 = torch.empty(M, N_half, device=device, dtype=dtype)
    
    xnumel = M * N_half
    grid = (triton.cdiv(xnumel, XBLOCK),)
    
    triton_fused_gating_activation_backward[grid](
        dg, h1, h0, a,
        dh0, dh1,
        xnumel,
        XBLOCK=XBLOCK,
    )
    
    # Step 3: dx = dh0 @ W_gate + dh1 @ W_value (extern_kernels.mm + addmm)
    dx = torch.mm(dh0, w_gate)           # (M, N_half) @ (N_half, K) = (M, K)
    dx = torch.addmm(dx, dh1, w_value)   # dx += dh1 @ w_value
    
    # Step 4: dW_gate = dh0^T @ X (extern_kernels.mm)
    dw_gate = torch.mm(dh0.T, x)  # (N_half, M) @ (M, K) = (N_half, K)
    
    # Step 5: dW_value = dh1^T @ X (extern_kernels.mm)
    dw_value = torch.mm(dh1.T, x)  # (N_half, M) @ (M, K) = (N_half, K)
    
    # Step 6: dW_down = g^T @ dy (extern_kernels.mm)
    dw_down = torch.mm(g.T, dy)  # (N_half, M) @ (M, K) = (N_half, K)
    
    return dx, dw_gate, dw_value, dw_down


##############################################################################################################################################

import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from typing import Dict

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}

result_gold = {}

######################################## HELPERS for Eval ######################################## 

def calculate_backward_gbps(params: Dict, ms: float) -> float:
    """Calculate GB/s for backward pass."""
    M = params['M']
    N = params['N']
    K = params['K']
    dtype = dtype_mapping[params['dtype_str']]
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    
    N_half = N // 2
    # Inputs: dy(M,K), x(M,K), w_gate(N/2,K), w_value(N/2,K), w_down(N/2,K), h0(M,N/2), h1(M,N/2), a(M,N/2), g(M,N/2)
    # Outputs: dx(M,K), dw_gate(N/2,K), dw_value(N/2,K), dw_down(N/2,K)
    input_bytes = (M*K + M*K + N_half*K + N_half*K + N_half*K + M*N_half*4) * bytes_per_element
    output_bytes = (M*K + N_half*K*3) * bytes_per_element
    total_bytes = input_bytes + output_bytes
    
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps


def calculate_backward_tflops(params: Dict, ms: float) -> float:
    """Calculate TFLOPS for backward pass."""
    M = params['M']
    N = params['N']
    K = params['K']
    N_half = N // 2
    
    # GEMMs: dg(M,N/2,K), dw_down(N/2,M,K), dx(M,N/2,K)*2, dw_gate(N/2,M,K), dw_value(N/2,M,K)
    gemm_flops = 2 * M * N_half * K * 6  # 6 GEMMs
    # Element-wise: da, dh1, dh0 (with SiLU backward ~10 ops)
    elem_flops = M * N_half * 15
    
    total_flops = gemm_flops + elem_flops
    tflops = total_flops / (ms / 1000) / 1e12
    return tflops


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def reference_forward(x, w_up, w_down):
    """PyTorch reference forward pass."""
    import torch.nn.functional as F
    N = w_up.shape[0]
    w_gate = w_up[:N // 2, :]
    w_value = w_up[N // 2:, :]
    
    h0 = x @ w_gate.T
    h1 = x @ w_value.T
    a = F.silu(h0)
    g = a * h1
    y = g @ w_down
    
    return y, (x, w_gate, w_value, w_down, h0, h1, a, g)


def reference_backward(dy, cache):
    """PyTorch reference backward pass."""
    x, w_gate, w_value, w_down, h0, h1, a, g = cache
    
    # dg = dy @ W_down^T
    dg = dy @ w_down.T
    # dW_down = g^T @ dy
    dw_down = g.T @ dy
    # da = dg * h1, dh1 = dg * a
    da = dg * h1
    dh1 = dg * a
    # dh0 = da * SiLU'(h0)
    sig = torch.sigmoid(h0)
    dh0 = da * sig * (1.0 + h0 * (1.0 - sig))
    # dX = dh0 @ W_gate + dh1 @ W_value
    dx = dh0 @ w_gate + dh1 @ w_value
    # dW_gate = dh0^T @ X, dW_value = dh1^T @ X
    dw_gate = dh0.T @ x
    dw_value = dh1.T @ x
    
    return dx, dw_gate, dw_value, dw_down

######################################## HELPERS for Eval ########################################


@pytest.mark.parametrize('M,N,K,XBLOCK,dtype_str', [
    # Small tests
    (4, 64, 32, 1024, 'float32'),
    (8, 128, 64, 1024, 'float32'),
    (16, 256, 128, 1024, 'float32'),
    # Phi-3 scale
    (4, 16384, 3072, 1024, 'float32'),
    (8, 16384, 3072, 1024, 'float32'),
    # LLaMA-7B scale
    (4, 22016, 4096, 1024, 'float32'),
    (8, 22016, 4096, 1024, 'float32'),
])
def test_op(M, N, K, XBLOCK, dtype_str, request):
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    device = 'cuda'
    N_half = N // 2
    
    # Create inputs
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_up = torch.randn(N, K, device=device, dtype=dtype)
    w_down = torch.randn(N_half, K, device=device, dtype=dtype)
    dy = torch.randn(M, K, device=device, dtype=dtype)
    
    # Reference forward to get cache
    _, cache = reference_forward(x, w_up, w_down)
    x_ref, w_gate, w_value, w_down_ref, h0, h1, a, g = cache
    
    # Reference backward
    dx_ref, dw_gate_ref, dw_value_ref, dw_down_ref = reference_backward(dy, cache)
    
    # Triton backward (ported from torch.compile)
    dx_tri, dw_gate_tri, dw_value_tri, dw_down_tri = ff_a16w16_fused_gated_backward(
        dy, x, w_gate, w_value, w_down,
        h0, h1, a, g,
        XBLOCK=XBLOCK,
    )
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    # Save results
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = dx_tri.clone().detach().cpu()
    
    # Assert correctness - relaxed tolerances for large-scale numerical precision
    rtol, atol = 1e-2, 5e-2
    assert_close(dx_tri, dx_ref, rtol=rtol, atol=atol, check_dtype=False)
    assert_close(dw_gate_tri, dw_gate_ref, rtol=rtol, atol=atol, check_dtype=False)
    assert_close(dw_value_tri, dw_value_ref, rtol=rtol, atol=atol, check_dtype=False)
    assert_close(dw_down_tri, dw_down_ref, rtol=rtol, atol=atol, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "ff_a16w16_fused_gated_backward_perf"

@pytest.mark.parametrize('M,N,K,XBLOCK,dtype_str', [
    # Small
    (8, 128, 64, 1024, 'float32'),
    # Medium
    (16, 4096, 2048, 1024, 'float32'),
    # Phi-3 scale
    (4, 16384, 3072, 1024, 'float32'),
    (8, 16384, 3072, 1024, 'float32'),
    # LLaMA-7B scale
    (4, 22016, 4096, 1024, 'float32'),
])
def test_performance(M, N, K, XBLOCK, dtype_str, request):
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    device = 'cuda'
    N_half = N // 2
    
    # Create inputs
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_up = torch.randn(N, K, device=device, dtype=dtype)
    w_down = torch.randn(N_half, K, device=device, dtype=dtype)
    dy = torch.randn(M, K, device=device, dtype=dtype)
    
    # Get cache from forward
    _, cache = reference_forward(x, w_up, w_down)
    _, w_gate, w_value, _, h0, h1, a, g = cache
    
    # Import benchmarker if available
    try:
        from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
        
        # Lambda for benchmarking
        op_lambda = lambda: ff_a16w16_fused_gated_backward(
            dy, x, w_gate, w_value, w_down,
            h0, h1, a, g,
            XBLOCK=XBLOCK,
        )
        
        bench_config = do_bench_config(warm_up=25, repetition=100)
        benchmarker = PytestBenchmarker(
            op_callable=op_lambda,
            op_name=OP_NAME_FOR_BENCHMARK,
            config=bench_config
        )
        
        current_params = {"M": M, "N": N, "K": K, "dtype_str": dtype_str}
        
        benchmarker.run_benchmark(
            current_params_dict=current_params,
            gbps_calculator=calculate_backward_gbps,
            tflops_calculator=calculate_backward_tflops
        )
    except ImportError:
        # Fallback: simple timing
        import time
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(10):
            ff_a16w16_fused_gated_backward(dy, x, w_gate, w_value, w_down, h0, h1, a, g, XBLOCK=XBLOCK)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            ff_a16w16_fused_gated_backward(dy, x, w_gate, w_value, w_down, h0, h1, a, g, XBLOCK=XBLOCK)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100 * 1000  # ms
        
        print(f"\nM={M}, N={N}, K={K}: {elapsed:.3f} ms")


######################################## HELPERS for Eval ########################################     

def test_save_results():  
    """Called after whole test run finished."""
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """Called after the test_performance function finishes."""
    print('\nPytest session finishing... Saving benchmark results...')
    try:
        from tb_eval.perf.ROCm.performance_utils_pytest import save_all_benchmark_results
        output_directory = os.path.join(os.path.dirname(__file__), "perf")
        os.makedirs(output_directory, exist_ok=True)
        save_all_benchmark_results(output_directory)
        print(f"All benchmark results attempted to save to: {output_directory}")
    except ImportError:
        print("Benchmark utilities not available, skipping save.")

######################################## HELPERS for Eval ########################################
