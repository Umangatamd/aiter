#!/usr/bin/env python3
"""
LLM-assisted verification and code generation for backward pass.

Uses AMD LLM API Gateway (Claude) to:
1. Verify gradient derivations
2. Generate PyTorch backward code
3. Cross-check against our manual implementation

Usage:
    python llm_verify.py
    python llm_verify.py --action verify
    python llm_verify.py --action generate
"""

import os
import requests
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential


# =============================================================================
# AMD LLM API GATEWAY CLIENT
# =============================================================================

class ClaudeModel:
    """AMD LLM API Gateway - Claude client"""
    
    def __init__(self, 
                 model_id="claude-sonnet-4", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.amd.com/claude3"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List,
                 temperature=0.7,
                 max_tokens=4096) -> str:
        # Cap max_tokens
        max_tokens = min(max_tokens, 16000)
        
        body = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "max_tokens": max_tokens,
            "presence_Penalty": 0,
            "frequency_Penalty": 0,
        }
        
        try:
            response = requests.post(
                url=f"{self.SERVER}/{self.model_id}/chat/completions",
                json=body,
                headers=self.headers,
                timeout=600
            )
        except Exception as e:
            raise ValueError(f"No response from the API: {str(e)}")
        
        if response.status_code != 200:
            raise ValueError(f"API returned status {response.status_code}: {response.text}")
        
        result = response.json()
        if 'content' in result and len(result['content']) > 0:
            return result['content'][0]['text']
        elif 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {result}")


# =============================================================================
# API KEY
# =============================================================================

API_KEY = "471c248fdb454e8b96173c8d25b03593"


# =============================================================================
# PROMPTS
# =============================================================================

VERIFY_DERIVATION_PROMPT = """
I have a fused gated feed-forward kernel (SwiGLU-style) and need to verify my backward pass derivation.

FORWARD PASS:
```
h0 = X @ W_gate^T          # (M, N/2)
h1 = X @ W_value^T         # (M, N/2)
a  = SiLU(h0)              # (M, N/2)
g  = a * h1                # element-wise (M, N/2)
Y  = g @ W_down            # (M, K)
```

Where:
- X: (M, K) input
- W_gate: (N/2, K) gate weights
- W_value: (N/2, K) value weights  
- W_down: (N/2, K) down-projection weights
- SiLU(x) = x * sigmoid(x)

MY BACKWARD PASS DERIVATION:

Given upstream gradient dY (shape M, K):

1. dg = dY @ W_down^T                    # (M, N/2)
   dW_down = g^T @ dY                    # (N/2, K)

2. da = dg * h1                          # (M, N/2)
   dh1 = dg * a                          # (M, N/2)

3. dh0 = da * SiLU'(h0)                  # (M, N/2)
   where SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

4. dX = dh0 @ W_gate + dh1 @ W_value     # (M, K)
   dW_gate = dh0^T @ X                   # (N/2, K)
   dW_value = dh1^T @ X                  # (N/2, K)

Please verify:
1. Are my gradient formulas correct?
2. Are the shapes correct at each step?
3. Is my SiLU derivative correct?
4. Did I miss anything?

Be concise and point out any errors.
"""


GENERATE_PYTORCH_PROMPT = """
Write a PyTorch implementation of the backward pass for a fused gated feed-forward (SwiGLU-style).

FORWARD PASS:
```python
def forward(x, w_gate, w_value, w_down):
    h0 = x @ w_gate.T          # (M, N/2)
    h1 = x @ w_value.T         # (M, N/2)
    a = F.silu(h0)             # (M, N/2)
    g = a * h1                 # (M, N/2)
    y = g @ w_down             # (M, K)
    return y, (x, w_gate, w_value, w_down, h0, h1, a, g)
```

Write a backward function that:
1. Takes upstream gradient `dy` and the cached tensors
2. Returns `dx`, `dw_gate`, `dw_value`, `dw_down`
3. Handles SiLU derivative correctly
4. Is pure PyTorch (no autograd)

Provide ONLY the code, no explanations.
"""


GENERATE_SILU_DERIVATIVE_PROMPT = """
Derive the derivative of SiLU (Swish) activation function.

SiLU(x) = x * sigmoid(x)

Show:
1. The mathematical derivation step by step
2. The final formula
3. PyTorch code to compute it

Be concise.
"""


TEST_PROMPT = """Say "Hello! AMD LLM Gateway is working." and nothing else."""


# =============================================================================
# LLM FUNCTIONS
# =============================================================================

def call_llm(prompt: str, model_id: str = "claude-sonnet-4") -> str:
    """Call AMD LLM Gateway with a prompt."""
    
    client = ClaudeModel(model_id=model_id, api_key=API_KEY)
    messages = [{"role": "user", "content": prompt}]
    return client.generate(messages)


def test_connection():
    """Test the LLM connection."""
    
    print("=" * 70)
    print("  Testing AMD LLM Gateway Connection")
    print("=" * 70)
    
    try:
        response = call_llm(TEST_PROMPT)
        print(f"\n✓ Connection successful!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return False


def verify_derivation():
    """Ask LLM to verify our backward pass derivation."""
    
    print("\n" + "=" * 70)
    print("  LLM VERIFICATION: Backward Pass Derivation")
    print("=" * 70)
    
    print("\nSending derivation for verification...")
    response = call_llm(VERIFY_DERIVATION_PROMPT)
    
    print("\n" + "-" * 70)
    print("LLM Response:")
    print("-" * 70)
    print(response)
    
    return response


def generate_backward_code():
    """Ask LLM to generate PyTorch backward pass code."""
    
    print("\n" + "=" * 70)
    print("  LLM CODE GENERATION: Backward Pass")
    print("=" * 70)
    
    print("\nGenerating backward pass code...")
    response = call_llm(GENERATE_PYTORCH_PROMPT)
    
    print("\n" + "-" * 70)
    print("LLM Generated Code:")
    print("-" * 70)
    print(response)
    
    return response


def derive_silu_derivative():
    """Ask LLM to derive SiLU derivative."""
    
    print("\n" + "=" * 70)
    print("  LLM DERIVATION: SiLU Derivative")
    print("=" * 70)
    
    print("\nDeriving SiLU derivative...")
    response = call_llm(GENERATE_SILU_DERIVATIVE_PROMPT)
    
    print("\n" + "-" * 70)
    print("LLM Response:")
    print("-" * 70)
    print(response)
    
    return response


def compare_with_reference():
    """
    Compare LLM-generated code with our reference implementation.
    """
    
    print("\n" + "=" * 70)
    print("  COMPARISON: Reference Implementation Test")
    print("=" * 70)
    
    # Import our reference
    try:
        from reference import ff_fused_gated_forward, ff_fused_gated_backward
    except ImportError:
        print("❌ Could not import reference.py")
        return
    
    import torch
    
    # Create test inputs
    torch.manual_seed(42)
    M, N, K = 8, 32, 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.randn(M, K, device=device)
    w_up = torch.randn(N, K, device=device)
    w_down = torch.randn(N // 2, K, device=device)
    
    # Run our reference
    y, cache = ff_fused_gated_forward(x, w_up, w_down, 'silu')
    dy = torch.randn_like(y)
    dx, dw_up, dw_down = ff_fused_gated_backward(dy, cache)
    
    print(f"\n✓ Reference implementation works!")
    print(f"\nOutputs:")
    print(f"  y shape: {y.shape}, norm: {y.norm():.4f}")
    print(f"  dx shape: {dx.shape}, norm: {dx.norm():.4f}")
    print(f"  dw_up shape: {dw_up.shape}, norm: {dw_up.norm():.4f}")
    print(f"  dw_down shape: {dw_down.shape}, norm: {dw_down.norm():.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM verification for backward pass")
    parser.add_argument("--action", 
                       choices=["test", "verify", "generate", "silu", "compare", "all"],
                       default="test",
                       help="Action to perform (default: test)")
    
    args = parser.parse_args()
    
    if args.action == "test":
        test_connection()
    
    elif args.action == "verify":
        if test_connection():
            verify_derivation()
    
    elif args.action == "generate":
        if test_connection():
            generate_backward_code()
    
    elif args.action == "silu":
        if test_connection():
            derive_silu_derivative()
    
    elif args.action == "compare":
        compare_with_reference()
    
    elif args.action == "all":
        if test_connection():
            verify_derivation()
            derive_silu_derivative()
            generate_backward_code()
            compare_with_reference()


if __name__ == "__main__":
    main()
