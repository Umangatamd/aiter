# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
PyTorch reference implementation for ff_a16w16_fused_gated forward and backward.

This serves as the ground truth for testing Triton backward kernels.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# =============================================================================
# ACTIVATION FUNCTIONS AND DERIVATIVES
# =============================================================================

def silu_forward(x: torch.Tensor) -> torch.Tensor:
    """SiLU/Swish: x * sigmoid(x)"""
    return F.silu(x)


def silu_backward(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    SiLU derivative: σ(x) * (1 + x * (1 - σ(x)))
    where σ(x) = sigmoid(x)
    """
    sig = torch.sigmoid(x)
    return grad * sig * (1.0 + x * (1.0 - sig))


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    return F.gelu(x)


def gelu_backward(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    GELU derivative: Φ(x) + x * φ(x)
    where Φ = CDF, φ = PDF of standard normal
    """
    sqrt_2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    
    cdf = 0.5 * (1.0 + torch.erf(x / sqrt_2))
    pdf = torch.exp(-0.5 * x * x) / sqrt_2pi
    return grad * (cdf + x * pdf)


def gelu_tanh_forward(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation"""
    return F.gelu(x, approximate='tanh')


def gelu_tanh_backward(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation derivative (use autograd)"""
    x_detached = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = F.gelu(x_detached, approximate='tanh')
        y.backward(grad)
    return x_detached.grad


def relu_forward(x: torch.Tensor) -> torch.Tensor:
    """ReLU: max(0, x)"""
    return F.relu(x)


def relu_backward(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """ReLU derivative: 1 if x > 0 else 0"""
    return grad * (x > 0).to(grad.dtype)


def identity_forward(x: torch.Tensor) -> torch.Tensor:
    """No activation"""
    return x


def identity_backward(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Identity derivative: 1"""
    return grad


# Activation dispatch
ACTIVATIONS = {
    'silu': (silu_forward, silu_backward),
    'gelu': (gelu_forward, gelu_backward),
    'gelu_tanh': (gelu_tanh_forward, gelu_tanh_backward),
    'relu': (relu_forward, relu_backward),
    None: (identity_forward, identity_backward),
}


def get_activation(name: Optional[str]):
    """Get forward and backward functions for activation."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]


# =============================================================================
# FORWARD PASS
# =============================================================================

def ff_fused_gated_forward(
    x: torch.Tensor,           # (M, K)
    w_up: torch.Tensor,        # (N, K) where N is even
    w_down: torch.Tensor,      # (N/2, K)
    activation: Optional[str] = 'silu'
) -> Tuple[torch.Tensor, Dict]:
    """
    Forward pass for fused gated feed-forward (SwiGLU-style).
    
    Computation:
        h0 = X @ W_gate^T           (M, N/2)
        h1 = X @ W_up^T             (M, N/2)
        a = activation(h0)          (M, N/2)
        g = a * h1                  (M, N/2)
        Y = g @ W_down              (M, K)
    
    Args:
        x: Input tensor (M, K)
        w_up: Up-projection weights (N, K), first half is gate, second half is value
        w_down: Down-projection weights (N/2, K)
        activation: Activation function name ('silu', 'gelu', 'relu', etc.)
    
    Returns:
        y: Output tensor (M, K)
        cache: Dictionary of saved tensors for backward pass
    """
    M, K = x.shape
    N = w_up.shape[0]
    
    assert N % 2 == 0, f"N must be even for gating, got {N}"
    assert w_up.shape[1] == K, f"w_up shape mismatch: {w_up.shape}"
    assert w_down.shape == (N // 2, K), f"w_down shape mismatch: {w_down.shape}"
    
    # Split W_up into gate and value weights
    w_gate = w_up[:N // 2, :]     # (N/2, K)
    w_value = w_up[N // 2:, :]    # (N/2, K)
    
    # Up-projection
    h0 = x @ w_gate.T             # (M, K) @ (K, N/2) = (M, N/2)
    h1 = x @ w_value.T            # (M, K) @ (K, N/2) = (M, N/2)
    
    # Activation on gate
    act_fwd, _ = get_activation(activation)
    a = act_fwd(h0)               # (M, N/2)
    
    # Gating (element-wise)
    g = a * h1                    # (M, N/2)
    
    # Down-projection
    y = g @ w_down                # (M, N/2) @ (N/2, K) = (M, K)
    
    # Cache for backward
    cache = {
        'x': x,
        'w_gate': w_gate,
        'w_value': w_value,
        'w_down': w_down,
        'h0': h0,           # Pre-activation (for activation derivative)
        'h1': h1,           # Value branch (for da = dg * h1)
        'a': a,             # Post-activation (for dh1 = dg * a)
        'g': g,             # Gated output (for dW_down)
        'activation': activation,
    }
    
    return y, cache


# =============================================================================
# BACKWARD PASS
# =============================================================================

def ff_fused_gated_backward(
    dy: torch.Tensor,          # (M, K) - upstream gradient
    cache: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for fused gated feed-forward.
    
    Args:
        dy: Upstream gradient (M, K)
        cache: Saved tensors from forward pass
    
    Returns:
        dx: Gradient w.r.t. input (M, K)
        dw_up: Gradient w.r.t. up-projection weights (N, K)
        dw_down: Gradient w.r.t. down-projection weights (N/2, K)
    """
    # Unpack cache
    x = cache['x']
    w_gate = cache['w_gate']
    w_value = cache['w_value']
    w_down = cache['w_down']
    h0 = cache['h0']
    h1 = cache['h1']
    a = cache['a']
    g = cache['g']
    activation = cache['activation']
    
    # Get activation backward function
    _, act_bwd = get_activation(activation)
    
    # =========================================================================
    # STEP 1: Backward through down-projection
    # Y = g @ W_down
    # =========================================================================
    dg = dy @ w_down.T            # (M, K) @ (K, N/2) = (M, N/2)
    dw_down = g.T @ dy            # (N/2, M) @ (M, K) = (N/2, K)
    
    # =========================================================================
    # STEP 2: Backward through gating
    # g = a * h1 (element-wise)
    # =========================================================================
    da = dg * h1                  # (M, N/2)
    dh1 = dg * a                  # (M, N/2)
    
    # =========================================================================
    # STEP 3: Backward through activation
    # a = activation(h0)
    # =========================================================================
    dh0 = act_bwd(h0, da)         # (M, N/2)
    
    # =========================================================================
    # STEP 4: Backward through up-projection
    # h0 = X @ W_gate^T
    # h1 = X @ W_value^T
    # =========================================================================
    dx_from_h0 = dh0 @ w_gate     # (M, N/2) @ (N/2, K) = (M, K)
    dx_from_h1 = dh1 @ w_value    # (M, N/2) @ (N/2, K) = (M, K)
    dx = dx_from_h0 + dx_from_h1  # (M, K)
    
    dw_gate = dh0.T @ x           # (N/2, M) @ (M, K) = (N/2, K)
    dw_value = dh1.T @ x          # (N/2, M) @ (M, K) = (N/2, K)
    
    # Concatenate to match w_up shape
    dw_up = torch.cat([dw_gate, dw_value], dim=0)  # (N, K)
    
    return dx, dw_up, dw_down


# =============================================================================
# AUTOGRAD FUNCTION WRAPPER
# =============================================================================

class FFGatedFunction(torch.autograd.Function):
    """
    Custom autograd function for fused gated feed-forward.
    Use this to integrate with PyTorch's autograd system.
    """
    
    @staticmethod
    def forward(ctx, x, w_up, w_down, activation='silu'):
        y, cache = ff_fused_gated_forward(x, w_up, w_down, activation)
        
        # Save tensors for backward
        ctx.save_for_backward(
            cache['x'], cache['w_gate'], cache['w_value'], cache['w_down'],
            cache['h0'], cache['h1'], cache['a'], cache['g']
        )
        ctx.activation = activation
        
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, w_gate, w_value, w_down, h0, h1, a, g = ctx.saved_tensors
        
        # Reconstruct cache
        cache = {
            'x': x, 'w_gate': w_gate, 'w_value': w_value, 'w_down': w_down,
            'h0': h0, 'h1': h1, 'a': a, 'g': g,
            'activation': ctx.activation
        }
        
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        return dx, dw_up, dw_down, None  # None for activation (not a tensor)


def ff_fused_gated(x, w_up, w_down, activation='silu'):
    """
    Functional interface using custom autograd.
    """
    return FFGatedFunction.apply(x, w_up, w_down, activation)

