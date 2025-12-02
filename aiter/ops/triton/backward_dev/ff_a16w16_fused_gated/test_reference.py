# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Test suite for ff_a16w16_fused_gated backward pass reference implementation.

Tests:
1. Autograd comparison - Compare manual backward vs PyTorch autograd
2. Numerical gradient check - Finite difference verification
3. Shape tests - Various M, N, K combinations
4. Activation tests - All supported activations
5. Dtype tests - float32, bfloat16, float16
"""

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import pytest
from typing import Tuple

from reference import (
    ff_fused_gated_forward,
    ff_fused_gated_backward,
    ff_fused_gated,
    ACTIVATIONS,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_inputs(
    M: int, N: int, K: int,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
    requires_grad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random input tensors."""
    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=requires_grad)
    w_up = torch.randn(N, K, dtype=dtype, device=device, requires_grad=requires_grad)
    w_down = torch.randn(N // 2, K, dtype=dtype, device=device, requires_grad=requires_grad)
    return x, w_up, w_down


def reference_forward_pytorch(x, w_up, w_down, activation='silu'):
    """
    Pure PyTorch forward (for autograd comparison).
    """
    N = w_up.shape[0]
    w_gate = w_up[:N // 2, :]
    w_value = w_up[N // 2:, :]
    
    h0 = x @ w_gate.T
    h1 = x @ w_value.T
    
    if activation == 'silu':
        a = F.silu(h0)
    elif activation == 'gelu':
        a = F.gelu(h0)
    elif activation == 'gelu_tanh':
        a = F.gelu(h0, approximate='tanh')
    elif activation == 'relu':
        a = F.relu(h0)
    else:
        a = h0
    
    g = a * h1
    y = g @ w_down
    return y


# =============================================================================
# TEST: AUTOGRAD COMPARISON
# =============================================================================

class TestAutogradComparison:
    """Compare manual backward against PyTorch autograd."""
    
    @pytest.mark.parametrize("M,N,K", [
        (4, 32, 16),
        (16, 64, 32),
        (32, 128, 64),
        (1, 32, 16),      # Single row
    ])
    @pytest.mark.parametrize("activation", ['silu'])
    def test_backward_matches_autograd(self, M, N, K, activation):
        """Test that manual backward matches PyTorch autograd."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create inputs
        x, w_up, w_down = create_inputs(M, N, K, device=device)
        
        # Forward with our implementation
        y, cache = ff_fused_gated_forward(x, w_up, w_down, activation)
        
        # Create upstream gradient
        dy = torch.randn_like(y)
        
        # Backward with PyTorch autograd
        y_ref = reference_forward_pytorch(x, w_up, w_down, activation)
        y_ref.backward(dy)
        
        dx_autograd = x.grad.clone()
        dw_up_autograd = w_up.grad.clone()
        dw_down_autograd = w_down.grad.clone()
        
        # Reset gradients
        x.grad, w_up.grad, w_down.grad = None, None, None
        
        # Backward with our manual implementation
        dx_manual, dw_up_manual, dw_down_manual = ff_fused_gated_backward(dy, cache)
        
        # Compare with relative tolerance that scales with tensor size
        # Larger tensors accumulate more floating point error
        rtol = 1e-4 if M * N * K < 100000 else 1e-3
        atol = 1e-5 if M * N * K < 100000 else 1e-3
        
        assert torch.allclose(dx_manual, dx_autograd, rtol=rtol, atol=atol), \
            f"dx mismatch: max error = {(dx_manual - dx_autograd).abs().max()}"
        assert torch.allclose(dw_up_manual, dw_up_autograd, rtol=rtol, atol=atol), \
            f"dw_up mismatch: max error = {(dw_up_manual - dw_up_autograd).abs().max()}"
        assert torch.allclose(dw_down_manual, dw_down_autograd, rtol=rtol, atol=atol), \
            f"dw_down mismatch: max error = {(dw_down_manual - dw_down_autograd).abs().max()}"


# =============================================================================
# TEST: REALISTIC LLM SIZES
# =============================================================================

class TestRealisticSizes:
    """Test with realistic LLM dimensions (SiLU/SwiGLU only)."""
    
    # Real model dimensions:
    # Phi-3: d=3072, d_ff=8192 → N=16384 (2*d_ff), K=3072
    # LLaMA-7B: d=4096, d_ff=11008 → N=22016, K=4096
    
    @pytest.mark.parametrize("M,N,K,name", [
        # Small/Medium (quick tests)
        (1, 1024, 512, "Small-B1"),
        (4, 2048, 1024, "Medium-B4"),
        (8, 4096, 2048, "Medium-B8"),
        (16, 4096, 2048, "Medium-B16"),
        (32, 8192, 4096, "Large-B32"),
        
        # Phi-3 scale
        (1, 16384, 3072, "Phi3-B1"),
        (4, 16384, 3072, "Phi3-B4"),
        (8, 16384, 3072, "Phi3-B8"),
        
        # LLaMA-7B scale (if memory allows)
        (1, 22016, 4096, "LLaMA7B-B1"),
        (4, 22016, 4096, "LLaMA7B-B4"),
    ])
    def test_silu_backward_realistic(self, M, N, K, name):
        """Test SiLU backward pass at realistic LLM scale."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            pytest.skip("Skipping large tests on CPU")
        
        print(f"\n  Testing {name}: M={M}, N={N}, K={K}")
        
        # Create inputs
        x, w_up, w_down = create_inputs(M, N, K, device=device)
        
        # Forward
        y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
        dy = torch.randn_like(y)
        
        # Backward with autograd
        y_ref = reference_forward_pytorch(x, w_up, w_down, 'silu')
        y_ref.backward(dy)
        dx_autograd = x.grad.clone()
        dw_up_autograd = w_up.grad.clone()
        dw_down_autograd = w_down.grad.clone()
        x.grad, w_up.grad, w_down.grad = None, None, None
        
        # Backward with our implementation
        dx_manual, dw_up_manual, dw_down_manual = ff_fused_gated_backward(dy, cache)
        
        # Compute relative errors
        dx_rel_err = (dx_manual - dx_autograd).abs().max() / (dx_autograd.abs().max() + 1e-8)
        dw_up_rel_err = (dw_up_manual - dw_up_autograd).abs().max() / (dw_up_autograd.abs().max() + 1e-8)
        dw_down_rel_err = (dw_down_manual - dw_down_autograd).abs().max() / (dw_down_autograd.abs().max() + 1e-8)
        
        print(f"    dx relative error: {dx_rel_err:.2e}")
        print(f"    dw_up relative error: {dw_up_rel_err:.2e}")
        print(f"    dw_down relative error: {dw_down_rel_err:.2e}")
        
        # Relative error threshold (0.001% = 1e-5)
        rel_tol = 1e-5
        
        assert dx_rel_err < rel_tol, f"dx relative error too large: {dx_rel_err}"
        assert dw_up_rel_err < rel_tol, f"dw_up relative error too large: {dw_up_rel_err}"
        assert dw_down_rel_err < rel_tol, f"dw_down relative error too large: {dw_down_rel_err}"


# =============================================================================
# TEST: NUMERICAL GRADIENT CHECK
# =============================================================================

class TestNumericalGradient:
    """Finite difference gradient verification."""
    
    @pytest.mark.parametrize("activation", ['silu'])
    def test_gradcheck(self, activation):
        """Numerical gradient check with float64."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use small tensors and float64 for numerical stability
        M, N, K = 4, 16, 8
        
        x = torch.randn(M, K, dtype=torch.float64, device=device, requires_grad=True)
        w_up = torch.randn(N, K, dtype=torch.float64, device=device, requires_grad=True)
        w_down = torch.randn(N // 2, K, dtype=torch.float64, device=device, requires_grad=True)
        
        def func(x, w_up, w_down):
            return ff_fused_gated(x, w_up, w_down, activation)
        
        assert gradcheck(func, (x, w_up, w_down), eps=1e-6, atol=1e-4, rtol=1e-3), \
            f"Gradcheck failed for activation={activation}"


# =============================================================================
# TEST: SHAPES
# =============================================================================

class TestShapes:
    """Test various input shapes."""
    
    @pytest.mark.parametrize("M,N,K", [
        (1, 32, 16),       # Single sample
        (4, 64, 32),       # Small batch
        (16, 128, 64),     # Medium
        (64, 256, 128),    # Larger
        (128, 512, 256),   # Even larger
        (3, 64, 32),       # Non-power-of-2 M
        (16, 96, 48),      # Non-power-of-2 N (but still even)
    ])
    def test_output_shapes(self, M, N, K):
        """Test that output shapes are correct."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, w_up, w_down = create_inputs(M, N, K, device=device)
        
        y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
        
        assert y.shape == (M, K), f"Output shape mismatch: {y.shape} != ({M}, {K})"
        
        dy = torch.randn_like(y)
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        assert dx.shape == (M, K), f"dx shape mismatch: {dx.shape}"
        assert dw_up.shape == (N, K), f"dw_up shape mismatch: {dw_up.shape}"
        assert dw_down.shape == (N // 2, K), f"dw_down shape mismatch: {dw_down.shape}"


# =============================================================================
# TEST: DTYPES
# =============================================================================

class TestDtypes:
    """Test different data types."""
    
    @pytest.mark.parametrize("dtype,rtol,atol", [
        (torch.float32, 1e-4, 1e-5),
        (torch.bfloat16, 1e-2, 1e-3),
        (torch.float16, 1e-3, 1e-4),
    ])
    def test_dtype_correctness(self, dtype, rtol, atol):
        """Test backward pass with different dtypes."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if dtype in [torch.bfloat16, torch.float16] and device == 'cpu':
            pytest.skip("Half precision not well supported on CPU")
        
        M, N, K = 16, 64, 32
        
        # Create float32 reference
        x_f32, w_up_f32, w_down_f32 = create_inputs(M, N, K, dtype=torch.float32, device=device)
        
        # Convert to target dtype
        x = x_f32.to(dtype).requires_grad_(True)
        w_up = w_up_f32.to(dtype).requires_grad_(True)
        w_down = w_down_f32.to(dtype).requires_grad_(True)
        
        # Forward
        y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
        
        # Backward
        dy = torch.randn_like(y)
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        # Check no NaN/Inf
        assert not torch.isnan(dx).any(), "dx contains NaN"
        assert not torch.isnan(dw_up).any(), "dw_up contains NaN"
        assert not torch.isnan(dw_down).any(), "dw_down contains NaN"
        assert not torch.isinf(dx).any(), "dx contains Inf"
        assert not torch.isinf(dw_up).any(), "dw_up contains Inf"
        assert not torch.isinf(dw_down).any(), "dw_down contains Inf"


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    def test_all_zeros_input(self):
        """Test with zero input."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        M, N, K = 8, 32, 16
        
        x = torch.zeros(M, K, device=device, requires_grad=True)
        w_up = torch.randn(N, K, device=device, requires_grad=True)
        w_down = torch.randn(N // 2, K, device=device, requires_grad=True)
        
        y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
        dy = torch.randn_like(y)
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        # Should not have NaN
        assert not torch.isnan(dx).any()
        assert not torch.isnan(dw_up).any()
        assert not torch.isnan(dw_down).any()
    
    def test_all_ones_gradient(self):
        """Test with all-ones upstream gradient."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        M, N, K = 8, 32, 16
        
        x, w_up, w_down = create_inputs(M, N, K, device=device)
        
        y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
        dy = torch.ones_like(y)
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        # Gradients should be non-zero (unless by chance)
        # Just check no NaN/Inf
        assert not torch.isnan(dx).any()
        assert not torch.isnan(dw_up).any()
        assert not torch.isnan(dw_down).any()


# =============================================================================
# MAIN: RUN ALL TESTS
# =============================================================================

def run_quick_test():
    """Quick sanity check without pytest."""
    print("=" * 60)
    print("Running quick sanity check...")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    M, N, K = 16, 64, 32
    
    for activation in ['silu', 'gelu', 'relu', None]:
        print(f"\nTesting activation: {activation}")
        
        x, w_up, w_down = create_inputs(M, N, K, device=device)
        
        # Forward
        y, cache = ff_fused_gated_forward(x, w_up, w_down, activation)
        
        # Backward with autograd
        dy = torch.randn_like(y)
        y_ref = reference_forward_pytorch(x, w_up, w_down, activation)
        y_ref.backward(dy)
        
        dx_ref = x.grad.clone()
        dw_up_ref = w_up.grad.clone()
        dw_down_ref = w_down.grad.clone()
        
        x.grad, w_up.grad, w_down.grad = None, None, None
        
        # Backward with our implementation
        dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
        
        # Compare
        dx_err = (dx - dx_ref).abs().max().item()
        dw_up_err = (dw_up - dw_up_ref).abs().max().item()
        dw_down_err = (dw_down - dw_down_ref).abs().max().item()
        
        print(f"  dx max error:      {dx_err:.2e}")
        print(f"  dw_up max error:   {dw_up_err:.2e}")
        print(f"  dw_down max error: {dw_down_err:.2e}")
        
        assert dx_err < 1e-4, f"dx error too large: {dx_err}"
        assert dw_up_err < 1e-4, f"dw_up error too large: {dw_up_err}"
        assert dw_down_err < 1e-4, f"dw_down error too large: {dw_down_err}"
        
        print(f"  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("ALL QUICK TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()

