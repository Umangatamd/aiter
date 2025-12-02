# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Triton backward kernel for ff_a16w16_fused_gated.

This is the INITIAL implementation - correct but not optimized.
Feed this to OpenEvolve for optimization.

Backward pass for:
    Y = (SiLU(X @ W_gate^T) * (X @ W_value^T)) @ W_down
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# ACTIVATION BACKWARD KERNELS
# =============================================================================

@triton.jit
def _silu_backward(x, grad):
    """
    SiLU backward: grad * σ(x) * (1 + x * (1 - σ(x)))
    """
    sig = tl.sigmoid(x)
    return grad * sig * (1.0 + x * (1.0 - sig))


# =============================================================================
# BACKWARD KERNEL: dg, dW_down
# =============================================================================

@triton.jit
def _backward_dg_dw_down_kernel(
    # Inputs
    dy_ptr, w_down_ptr, g_ptr,
    # Outputs
    dg_ptr, dw_down_ptr,
    # Dimensions
    M, N_half, K,
    # Strides for dy (M, K)
    stride_dy_m, stride_dy_k,
    # Strides for w_down (N_half, K)
    stride_wd_n, stride_wd_k,
    # Strides for g (M, N_half)
    stride_g_m, stride_g_n,
    # Strides for dg (M, N_half)
    stride_dg_m, stride_dg_n,
    # Strides for dw_down (N_half, K)
    stride_dwd_n, stride_dwd_k,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute:
        dg = dy @ W_down^T        (M, K) @ (K, N_half) = (M, N_half)
        dW_down = g^T @ dy        (N_half, M) @ (M, K) = (N_half, K)
    
    This kernel computes dg. dW_down is computed separately.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Accumulator for dg block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # dg = dy @ W_down^T
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        # Load dy block: (BLOCK_M, BLOCK_K)
        dy_ptrs = dy_ptr + offs_m[:, None] * stride_dy_m + k_offs[None, :] * stride_dy_k
        dy_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        
        # Load W_down^T block: (BLOCK_K, BLOCK_N) = W_down transposed
        # W_down is (N_half, K), W_down^T is (K, N_half)
        wd_ptrs = w_down_ptr + offs_n[None, :] * stride_wd_n + k_offs[:, None] * stride_wd_k
        wd_mask = (offs_n[None, :] < N_half) & (k_offs[:, None] < K)
        wd_t = tl.load(wd_ptrs, mask=wd_mask, other=0.0)
        
        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(dy, wd_t)
    
    # Store dg
    dg_ptrs = dg_ptr + offs_m[:, None] * stride_dg_m + offs_n[None, :] * stride_dg_n
    dg_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_half)
    tl.store(dg_ptrs, acc.to(dg_ptr.dtype.element_ty), mask=dg_mask)


@triton.jit
def _backward_dw_down_kernel(
    # Inputs
    g_ptr, dy_ptr,
    # Output
    dw_down_ptr,
    # Dimensions
    M, N_half, K,
    # Strides
    stride_g_m, stride_g_n,
    stride_dy_m, stride_dy_k,
    stride_dwd_n, stride_dwd_k,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: dW_down = g^T @ dy    (N_half, M) @ (M, K) = (N_half, K)
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_M)):
        m_offs = m * BLOCK_M + offs_m
        
        # Load g^T block: (BLOCK_N, BLOCK_M)
        g_ptrs = g_ptr + m_offs[None, :] * stride_g_m + offs_n[:, None] * stride_g_n
        g_mask = (m_offs[None, :] < M) & (offs_n[:, None] < N_half)
        g_t = tl.load(g_ptrs, mask=g_mask, other=0.0)
        
        # Load dy block: (BLOCK_M, BLOCK_K)
        dy_ptrs = dy_ptr + m_offs[:, None] * stride_dy_m + offs_k[None, :] * stride_dy_k
        dy_mask = (m_offs[:, None] < M) & (offs_k[None, :] < K)
        dy_block = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        
        # Accumulate: (BLOCK_N, BLOCK_M) @ (BLOCK_M, BLOCK_K)
        acc += tl.dot(g_t, dy_block)
    
    # Store dW_down
    dwd_ptrs = dw_down_ptr + offs_n[:, None] * stride_dwd_n + offs_k[None, :] * stride_dwd_k
    dwd_mask = (offs_n[:, None] < N_half) & (offs_k[None, :] < K)
    tl.store(dwd_ptrs, acc.to(dw_down_ptr.dtype.element_ty), mask=dwd_mask)


# =============================================================================
# BACKWARD KERNEL: da, dh1 (through gating)
# =============================================================================

@triton.jit
def _backward_gating_kernel(
    # Inputs
    dg_ptr, h1_ptr, a_ptr,
    # Outputs
    da_ptr, dh1_ptr,
    # Dimensions
    M, N_half,
    # Strides
    stride_m, stride_n,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute element-wise:
        da = dg * h1
        dh1 = dg * a
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M * N_half
    
    # Load
    dg = tl.load(dg_ptr + offs, mask=mask)
    h1 = tl.load(h1_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    
    # Compute
    da = dg * h1
    dh1 = dg * a
    
    # Store
    tl.store(da_ptr + offs, da, mask=mask)
    tl.store(dh1_ptr + offs, dh1, mask=mask)


# =============================================================================
# BACKWARD KERNEL: dh0 (through activation)
# =============================================================================

@triton.jit
def _backward_activation_kernel(
    # Inputs
    da_ptr, h0_ptr,
    # Output
    dh0_ptr,
    # Dimensions
    numel,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute: dh0 = da * SiLU'(h0)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    da = tl.load(da_ptr + offs, mask=mask)
    h0 = tl.load(h0_ptr + offs, mask=mask)
    
    # SiLU backward
    dh0 = _silu_backward(h0, da)
    
    tl.store(dh0_ptr + offs, dh0, mask=mask)


# =============================================================================
# BACKWARD KERNEL: dX, dW_gate, dW_value (through up-projection)
# =============================================================================

@triton.jit
def _backward_dx_kernel(
    # Inputs
    dh0_ptr, dh1_ptr, w_gate_ptr, w_value_ptr,
    # Output
    dx_ptr,
    # Dimensions
    M, N_half, K,
    # Strides for dh0, dh1 (M, N_half)
    stride_dh_m, stride_dh_n,
    # Strides for w_gate, w_value (N_half, K)
    stride_w_n, stride_w_k,
    # Strides for dx (M, K)
    stride_dx_m, stride_dx_k,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: dX = dh0 @ W_gate + dh1 @ W_value
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N_half, BLOCK_N)):
        n_offs = n * BLOCK_N + offs_n
        
        # Load dh0 block: (BLOCK_M, BLOCK_N)
        dh0_ptrs = dh0_ptr + offs_m[:, None] * stride_dh_m + n_offs[None, :] * stride_dh_n
        dh0_mask = (offs_m[:, None] < M) & (n_offs[None, :] < N_half)
        dh0 = tl.load(dh0_ptrs, mask=dh0_mask, other=0.0)
        
        # Load W_gate block: (BLOCK_N, BLOCK_K)
        wg_ptrs = w_gate_ptr + n_offs[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        wg_mask = (n_offs[:, None] < N_half) & (offs_k[None, :] < K)
        wg = tl.load(wg_ptrs, mask=wg_mask, other=0.0)
        
        # Accumulate dh0 @ W_gate
        acc += tl.dot(dh0, wg)
        
        # Load dh1 block: (BLOCK_M, BLOCK_N)
        dh1_ptrs = dh1_ptr + offs_m[:, None] * stride_dh_m + n_offs[None, :] * stride_dh_n
        dh1 = tl.load(dh1_ptrs, mask=dh0_mask, other=0.0)
        
        # Load W_value block: (BLOCK_N, BLOCK_K)
        wv_ptrs = w_value_ptr + n_offs[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        wv = tl.load(wv_ptrs, mask=wg_mask, other=0.0)
        
        # Accumulate dh1 @ W_value
        acc += tl.dot(dh1, wv)
    
    # Store dx
    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dx_m + offs_k[None, :] * stride_dx_k
    dx_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(dx_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=dx_mask)


@triton.jit
def _backward_dw_kernel(
    # Inputs
    dh_ptr, x_ptr,
    # Output
    dw_ptr,
    # Dimensions
    M, N_half, K,
    # Strides for dh (M, N_half)
    stride_dh_m, stride_dh_n,
    # Strides for x (M, K)
    stride_x_m, stride_x_k,
    # Strides for dw (N_half, K)
    stride_dw_n, stride_dw_k,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: dW = dh^T @ X    (N_half, M) @ (M, K) = (N_half, K)
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_M)):
        m_offs = m * BLOCK_M + offs_m
        
        # Load dh^T block: (BLOCK_N, BLOCK_M)
        dh_ptrs = dh_ptr + m_offs[None, :] * stride_dh_m + offs_n[:, None] * stride_dh_n
        dh_mask = (m_offs[None, :] < M) & (offs_n[:, None] < N_half)
        dh_t = tl.load(dh_ptrs, mask=dh_mask, other=0.0)
        
        # Load x block: (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + m_offs[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_mask = (m_offs[:, None] < M) & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Accumulate: (BLOCK_N, BLOCK_M) @ (BLOCK_M, BLOCK_K)
        acc += tl.dot(dh_t, x_block)
    
    # Store dw
    dw_ptrs = dw_ptr + offs_n[:, None] * stride_dw_n + offs_k[None, :] * stride_dw_k
    dw_mask = (offs_n[:, None] < N_half) & (offs_k[None, :] < K)
    tl.store(dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=dw_mask)


# =============================================================================
# PYTHON WRAPPER
# =============================================================================

def ff_a16w16_fused_gated_backward_triton(
    dy: torch.Tensor,          # (M, K)
    x: torch.Tensor,           # (M, K)
    w_gate: torch.Tensor,      # (N/2, K)
    w_value: torch.Tensor,     # (N/2, K)
    w_down: torch.Tensor,      # (N/2, K)
    h0: torch.Tensor,          # (M, N/2) - pre-activation
    h1: torch.Tensor,          # (M, N/2) - value branch
    a: torch.Tensor,           # (M, N/2) - post-activation
    g: torch.Tensor,           # (M, N/2) - gated output
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton backward pass for fused gated feed-forward.
    
    Returns: dx, dw_gate, dw_value, dw_down
    """
    M, K = dy.shape
    N_half = w_gate.shape[0]
    
    device = dy.device
    dtype = dy.dtype
    
    # Allocate outputs
    dg = torch.empty(M, N_half, device=device, dtype=dtype)
    dw_down = torch.empty(N_half, K, device=device, dtype=dtype)
    da = torch.empty(M, N_half, device=device, dtype=dtype)
    dh1 = torch.empty(M, N_half, device=device, dtype=dtype)
    dh0 = torch.empty(M, N_half, device=device, dtype=dtype)
    dx = torch.empty(M, K, device=device, dtype=dtype)
    dw_gate = torch.empty(N_half, K, device=device, dtype=dtype)
    dw_value = torch.empty(N_half, K, device=device, dtype=dtype)
    
    # Block sizes (can be tuned by OpenEvolve)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    BLOCK_SIZE = 1024
    
    # Step 1: dg = dy @ W_down^T
    grid_dg = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_half, BLOCK_N))
    _backward_dg_dw_down_kernel[grid_dg](
        dy, w_down, g,
        dg, dw_down,  # dw_down not computed here
        M, N_half, K,
        dy.stride(0), dy.stride(1),
        w_down.stride(0), w_down.stride(1),
        g.stride(0), g.stride(1),
        dg.stride(0), dg.stride(1),
        dw_down.stride(0), dw_down.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Step 2: dW_down = g^T @ dy
    grid_dwd = (triton.cdiv(N_half, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _backward_dw_down_kernel[grid_dwd](
        g, dy,
        dw_down,
        M, N_half, K,
        g.stride(0), g.stride(1),
        dy.stride(0), dy.stride(1),
        dw_down.stride(0), dw_down.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Step 3: da = dg * h1, dh1 = dg * a
    numel = M * N_half
    grid_gate = (triton.cdiv(numel, BLOCK_SIZE),)
    _backward_gating_kernel[grid_gate](
        dg, h1, a,
        da, dh1,
        M, N_half,
        dg.stride(0), dg.stride(1),
        BLOCK_SIZE,
    )
    
    # Step 4: dh0 = da * SiLU'(h0)
    _backward_activation_kernel[grid_gate](
        da, h0,
        dh0,
        numel,
        BLOCK_SIZE,
    )
    
    # Step 5: dX = dh0 @ W_gate + dh1 @ W_value
    grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _backward_dx_kernel[grid_dx](
        dh0, dh1, w_gate, w_value,
        dx,
        M, N_half, K,
        dh0.stride(0), dh0.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        dx.stride(0), dx.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Step 6: dW_gate = dh0^T @ X
    grid_dwg = (triton.cdiv(N_half, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _backward_dw_kernel[grid_dwg](
        dh0, x,
        dw_gate,
        M, N_half, K,
        dh0.stride(0), dh0.stride(1),
        x.stride(0), x.stride(1),
        dw_gate.stride(0), dw_gate.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Step 7: dW_value = dh1^T @ X
    _backward_dw_kernel[grid_dwg](
        dh1, x,
        dw_value,
        M, N_half, K,
        dh1.stride(0), dh1.stride(1),
        x.stride(0), x.stride(1),
        dw_value.stride(0), dw_value.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return dx, dw_gate, dw_value, dw_down


# =============================================================================
# CONVENIENCE WRAPPER (matches reference.py interface)
# =============================================================================

def ff_fused_gated_backward_triton(dy: torch.Tensor, cache: dict):
    """
    Wrapper that matches the reference.py interface.
    
    Args:
        dy: Upstream gradient (M, K)
        cache: Dict with saved tensors from forward
    
    Returns:
        dx, dw_up, dw_down
    """
    dx, dw_gate, dw_value, dw_down = ff_a16w16_fused_gated_backward_triton(
        dy,
        cache['x'],
        cache['w_gate'],
        cache['w_value'],
        cache['w_down'],
        cache['h0'],
        cache['h1'],
        cache['a'],
        cache['g'],
    )
    
    # Concatenate dw_gate and dw_value to match dw_up shape
    dw_up = torch.cat([dw_gate, dw_value], dim=0)
    
    return dx, dw_up, dw_down

