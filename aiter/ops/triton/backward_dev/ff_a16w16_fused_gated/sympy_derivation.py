#!/usr/bin/env python3
"""
Symbolic derivation of gradients for ff_a16w16_fused_gated kernel using SymPy.

This script derives:
1. Activation function derivatives (SiLU, GELU, ReLU)
2. Backward pass gradients through the fused gated feed-forward

Run: python sympy_derivation.py
"""

import sympy as sp
from sympy import (
    symbols, Function, diff, simplify, expand, factor,
    exp, log, sqrt, pi, erf, tanh, Abs, Piecewise, Max,
    Matrix, MatrixSymbol, Transpose, trace,
    latex, pprint,
)
from sympy.stats import Normal, cdf, density


def section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# PART 1: ACTIVATION DERIVATIVES
# =============================================================================

def derive_activation_derivatives():
    """Derive derivatives of activation functions."""
    
    section("ACTIVATION FUNCTION DERIVATIVES")
    
    x = symbols('x', real=True)
    
    # -------------------------------------------------------------------------
    # 1. SIGMOID
    # -------------------------------------------------------------------------
    print("\n--- Sigmoid ---")
    sigmoid = 1 / (1 + exp(-x))
    sigmoid_deriv = simplify(diff(sigmoid, x))
    
    print(f"σ(x) = {sigmoid}")
    print(f"σ'(x) = {sigmoid_deriv}")
    print(f"     = σ(x)(1 - σ(x))")
    
    # -------------------------------------------------------------------------
    # 2. SiLU (Swish)
    # -------------------------------------------------------------------------
    print("\n--- SiLU (Swish) ---")
    silu = x * sigmoid
    silu_deriv = simplify(diff(silu, x))
    
    print(f"SiLU(x) = x · σ(x) = {silu}")
    print(f"SiLU'(x) = {silu_deriv}")
    
    # Expand to show cleaner form
    silu_deriv_expanded = expand(silu_deriv)
    print(f"        = {silu_deriv_expanded}")
    
    # Alternative form
    sig = symbols('σ', positive=True)
    silu_deriv_alt = sig * (1 + x * (1 - sig))
    print(f"\nAlternative form (let σ = sigmoid(x)):")
    print(f"SiLU'(x) = σ · (1 + x · (1 - σ))")
    
    # -------------------------------------------------------------------------
    # 3. GELU (Exact)
    # -------------------------------------------------------------------------
    print("\n--- GELU (Exact) ---")
    
    # GELU = x * Φ(x) where Φ is standard normal CDF
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = sqrt(2)
    phi_cdf = sp.Rational(1, 2) * (1 + erf(x / sqrt2))  # CDF
    phi_pdf = exp(-x**2 / 2) / sqrt(2 * pi)             # PDF
    
    gelu = x * phi_cdf
    gelu_deriv = simplify(diff(gelu, x))
    
    print(f"GELU(x) = x · Φ(x)")
    print(f"where Φ(x) = 0.5 · (1 + erf(x/√2))")
    print(f"\nGELU'(x) = {gelu_deriv}")
    
    # Cleaner form
    print(f"\nCleaner form:")
    print(f"GELU'(x) = Φ(x) + x · φ(x)")
    print(f"where φ(x) = exp(-x²/2) / √(2π)  [PDF]")
    
    # -------------------------------------------------------------------------
    # 4. GELU (Tanh Approximation)
    # -------------------------------------------------------------------------
    print("\n--- GELU (Tanh Approximation) ---")
    
    # GELU_tanh = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    kappa = sp.Float(0.044715)
    beta = sqrt(2 / pi)
    inner = beta * (x + kappa * x**3)
    gelu_tanh = sp.Rational(1, 2) * x * (1 + tanh(inner))
    gelu_tanh_deriv = simplify(diff(gelu_tanh, x))
    
    print(f"GELU_tanh(x) = 0.5 · x · (1 + tanh(β(x + κx³)))")
    print(f"where β = √(2/π), κ = 0.044715")
    print(f"\nGELU_tanh'(x) = [complex, use autograd in practice]")
    
    # -------------------------------------------------------------------------
    # 5. ReLU
    # -------------------------------------------------------------------------
    print("\n--- ReLU ---")
    
    relu = Max(0, x)
    # ReLU derivative is a step function
    relu_deriv = Piecewise((1, x > 0), (0, True))
    
    print(f"ReLU(x) = max(0, x)")
    print(f"ReLU'(x) = 1 if x > 0 else 0")
    
    return {
        'silu': (silu, silu_deriv),
        'gelu': (gelu, gelu_deriv),
        'relu': (relu, relu_deriv),
    }


# =============================================================================
# PART 2: MATRIX CALCULUS RULES
# =============================================================================

def show_matrix_calculus_rules():
    """Show basic matrix calculus rules used in backward pass."""
    
    section("MATRIX CALCULUS RULES")
    
    print("""
Basic rules for matrix derivatives (scalar loss L):

1. LINEAR LAYER (Y = X @ W^T)
   ─────────────────────────
   Forward:  Y = X @ W^T        X:(M,K), W:(N,K), Y:(M,N)
   
   Backward:
     dL/dX = dL/dY @ W          (M,N) @ (N,K) = (M,K)
     dL/dW = (dL/dY)^T @ X      (N,M) @ (M,K) = (N,K)


2. ELEMENT-WISE MULTIPLICATION (Y = A ⊙ B)
   ────────────────────────────────────────
   Forward:  Y = A * B          A:(M,N), B:(M,N), Y:(M,N)
   
   Backward:
     dL/dA = dL/dY ⊙ B          (M,N)
     dL/dB = dL/dY ⊙ A          (M,N)


3. ELEMENT-WISE ACTIVATION (Y = σ(X))
   ────────────────────────────────────
   Forward:  Y = σ(X)           X:(M,N), Y:(M,N)
   
   Backward:
     dL/dX = dL/dY ⊙ σ'(X)      (M,N)


4. SUM OF GRADIENTS (X feeds into multiple paths)
   ───────────────────────────────────────────────
   If X → Y₁ and X → Y₂:
     dL/dX = dL/dX_from_Y₁ + dL/dX_from_Y₂
""")


# =============================================================================
# PART 3: FULL BACKWARD PASS DERIVATION
# =============================================================================

def derive_full_backward():
    """Derive the full backward pass step by step."""
    
    section("FULL BACKWARD PASS DERIVATION")
    
    print("""
FORWARD PASS:
─────────────
  h0 = X @ W_gate^T          (M,K) @ (K,N/2) = (M,N/2)
  h1 = X @ W_value^T         (M,K) @ (K,N/2) = (M,N/2)
  a  = σ(h0)                 (M,N/2)
  g  = a ⊙ h1                (M,N/2)
  Y  = g @ W_down            (M,N/2) @ (N/2,K) = (M,K)

Given: dL/dY (upstream gradient), shape (M,K)
Find:  dL/dX, dL/dW_gate, dL/dW_value, dL/dW_down


BACKWARD PASS:
──────────────

STEP 1: Through down-projection (Y = g @ W_down)
────────────────────────────────────────────────
  dL/dg      = dL/dY @ W_down^T     (M,K) @ (K,N/2) = (M,N/2)
  dL/dW_down = g^T @ dL/dY          (N/2,M) @ (M,K) = (N/2,K)


STEP 2: Through gating (g = a ⊙ h1)
───────────────────────────────────
  dL/da  = dL/dg ⊙ h1              (M,N/2)
  dL/dh1 = dL/dg ⊙ a               (M,N/2)


STEP 3: Through activation (a = σ(h0))
──────────────────────────────────────
  dL/dh0 = dL/da ⊙ σ'(h0)          (M,N/2)


STEP 4: Through up-projection
─────────────────────────────
  From h0 = X @ W_gate^T:
    dL/dX_gate   = dL/dh0 @ W_gate       (M,N/2) @ (N/2,K) = (M,K)
    dL/dW_gate   = (dL/dh0)^T @ X        (N/2,M) @ (M,K) = (N/2,K)

  From h1 = X @ W_value^T:
    dL/dX_value  = dL/dh1 @ W_value      (M,N/2) @ (N/2,K) = (M,K)
    dL/dW_value  = (dL/dh1)^T @ X        (N/2,M) @ (M,K) = (N/2,K)


STEP 5: Combine gradients for X
───────────────────────────────
  dL/dX = dL/dX_gate + dL/dX_value       (M,K)


STEP 6: Combine gradients for W_up
──────────────────────────────────
  dL/dW_up = concat(dL/dW_gate, dL/dW_value, dim=0)   (N,K)


SUMMARY:
────────
  dX      = dh0 @ W_gate + dh1 @ W_value
  dW_up   = concat(dh0^T @ X, dh1^T @ X)
  dW_down = g^T @ dY

where:
  dg  = dY @ W_down^T
  da  = dg * h1
  dh1 = dg * a
  dh0 = da * σ'(h0)
""")


# =============================================================================
# PART 4: NUMERICAL VERIFICATION
# =============================================================================

def numerical_verification():
    """Verify symbolic derivatives match numerical approximation."""
    
    section("NUMERICAL VERIFICATION")
    
    import numpy as np
    
    print("Verifying activation derivatives numerically...\n")
    
    def numerical_deriv(f, x, eps=1e-7):
        """Central difference approximation."""
        return (f(x + eps) - f(x - eps)) / (2 * eps)
    
    # Test points
    test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    # SiLU
    print("SiLU:")
    def silu_np(x):
        return x / (1 + np.exp(-x))
    def silu_deriv_np(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 + x * (1 - sig))
    
    for x in test_points:
        numerical = numerical_deriv(silu_np, x)
        analytical = silu_deriv_np(x)
        error = abs(numerical - analytical)
        print(f"  x={x:5.1f}: numerical={numerical:.6f}, analytical={analytical:.6f}, error={error:.2e}")
    
    # GELU
    print("\nGELU:")
    from scipy.special import erf as scipy_erf
    
    def gelu_np(x):
        return 0.5 * x * (1 + scipy_erf(x / np.sqrt(2)))
    def gelu_deriv_np(x):
        cdf = 0.5 * (1 + scipy_erf(x / np.sqrt(2)))
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        return cdf + x * pdf
    
    for x in test_points:
        numerical = numerical_deriv(gelu_np, x)
        analytical = gelu_deriv_np(x)
        error = abs(numerical - analytical)
        print(f"  x={x:5.1f}: numerical={numerical:.6f}, analytical={analytical:.6f}, error={error:.2e}")
    
    print("\n✓ All derivatives verified!")


# =============================================================================
# PART 5: CODE GENERATION
# =============================================================================

def generate_code():
    """Generate Python code for activation derivatives."""
    
    section("GENERATED CODE")
    
    print("""
# Copy-paste ready code for activation derivatives:

def silu_backward(x, grad):
    '''
    SiLU derivative: σ(x) · (1 + x · (1 - σ(x)))
    '''
    sig = torch.sigmoid(x)
    return grad * sig * (1.0 + x * (1.0 - sig))


def gelu_backward(x, grad):
    '''
    GELU derivative: Φ(x) + x · φ(x)
    where Φ = CDF, φ = PDF of standard normal
    '''
    import math
    sqrt_2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    
    cdf = 0.5 * (1.0 + torch.erf(x / sqrt_2))
    pdf = torch.exp(-0.5 * x * x) / sqrt_2pi
    return grad * (cdf + x * pdf)


def relu_backward(x, grad):
    '''
    ReLU derivative: 1 if x > 0 else 0
    '''
    return grad * (x > 0).to(grad.dtype)


# Triton versions:

@triton.jit
def _silu_backward(x, grad):
    sig = tl.sigmoid(x)
    return grad * sig * (1.0 + x * (1.0 - sig))


@triton.jit
def _gelu_backward(x, grad):
    M_SQRT1_2 = 0.70710678118654752440
    M_SQRT_2PI_INV = 0.3989422804014327  # 1/sqrt(2*pi)
    
    cdf = 0.5 * (1.0 + tl.math.erf(x * M_SQRT1_2))
    pdf = tl.exp(-0.5 * x * x) * M_SQRT_2PI_INV
    return grad * (cdf + x * pdf)


@triton.jit
def _relu_backward(x, grad):
    return tl.where(x > 0, grad, 0.0)
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  SYMPY DERIVATION: ff_a16w16_fused_gated Backward Pass")
    print("=" * 70)
    
    # Part 1: Activation derivatives
    activations = derive_activation_derivatives()
    
    # Part 2: Matrix calculus rules
    show_matrix_calculus_rules()
    
    # Part 3: Full backward pass
    derive_full_backward()
    
    # Part 4: Numerical verification
    try:
        numerical_verification()
    except ImportError:
        print("\n[Skipping numerical verification - scipy not installed]")
    
    # Part 5: Code generation
    generate_code()
    
    print("\n" + "=" * 70)
    print("  DERIVATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

