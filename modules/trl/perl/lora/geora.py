# copyright (c) 2025, mikastars39.org
# All rights reserved.
# This source code is licensed under the Apache-2.0 License.
# See the LICENSE file in the root directory for details.

"""
GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR
Reference: https://arxiv.org/abs/2601.09361

GeoRA is a PEFT method tailored for Reinforcement Learning with Verifiable Rewards (RLVR).
Key design:
  1. Constructs a geometry-constrained matrix W_Geo using Spectral + Euclidean mask priors.
  2. Performs SVD on W_Geo and initializes LoRA adapters from principal (top-r) components.
  3. Freezes a Residual Matrix W_res = W - (alpha/r) * B_Geo @ A_Geo as a stability anchor.
  4. Forward: h = W_res @ x + (alpha/r) * B_Geo @ A_Geo @ x  (function-preserving at init)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometric Prior Construction (Section 3.2)
# ---------------------------------------------------------------------------


def build_geometry_mask(
    weight: torch.Tensor, rank: int, sparsity_ratio: float = 0.2
) -> torch.Tensor:
    """
    Construct the geometry mask M_Geo = M_Spec ∪ M_Euc (Equation 6-8 in GeoRA paper).

    Args:
        weight:         Weight matrix W  (shape: [out, in])
        rank:           Rank r used for low-rank approximation W_hat_r
        sparsity_ratio: ρ — fraction of entries to select (default 0.2)

    Returns:
        M_Geo: Binary mask of same shape as weight (torch.bool)
    """
    W = weight.float()

    # --- Spectral Prior M_Spec: low-magnitude entries in rank-r approximation ---
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Rank-r approximation
    W_hat_r = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
    tau_spec = torch.quantile(W_hat_r.abs(), sparsity_ratio)
    M_spec = W_hat_r.abs() <= tau_spec  # (Equation 6)

    # --- Euclidean Prior M_Euc: low-magnitude entries in original weight ---
    tau_euc = torch.quantile(W.abs(), sparsity_ratio)
    M_euc = W.abs() <= tau_euc  # (Equation 7)

    # Union of two stable subspaces (Equation 8)
    M_geo = M_spec | M_euc
    return M_geo


def build_w_geo(
    weight: torch.Tensor, rank: int, sparsity_ratio: float = 0.2
) -> torch.Tensor:
    """
    Construct the geometry-constrained matrix W_Geo = W ⊙ M_Geo.

    Args:
        weight:         Weight matrix W  (shape: [out, in])
        rank:           Rank r for spectral prior
        sparsity_ratio: ρ

    Returns:
        W_Geo: Masked weight matrix (same shape as weight)
    """
    M_geo = build_geometry_mask(weight, rank, sparsity_ratio)
    W_geo = weight.float() * M_geo.float()
    return W_geo


# ---------------------------------------------------------------------------
# Adapter Initialization (Section 3.1)
# ---------------------------------------------------------------------------


def initialize_geora_layer(
    weight: torch.Tensor,
    rank: int,
    lora_alpha: int,
    sparsity_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GeoRA initialization from geometry-constrained SVD (Equations 1-4).

    Steps:
      1. Build W_Geo = W ⊙ M_Geo
      2. SVD: W_Geo = U_Geo Σ_Geo V_Geo^T
      3. Extract top-r components → A_Geo, B_Geo  (Equations 2-3)
      4. Compute Residual: W_res = W - (alpha/r) * B_Geo @ A_Geo  (Equation 4)

    Args:
        weight:         Pre-trained weight matrix  [out_features, in_features]
        rank:           LoRA rank r
        lora_alpha:     LoRA scaling α
        sparsity_ratio: ρ for geometric mask

    Returns:
        (A_Geo, B_Geo, W_res)
          A_Geo:  [rank, in_features]   — lora_A initialization
          B_Geo:  [out_features, rank]  — lora_B initialization
          W_res:  [out_features, in_features] — frozen residual
    """
    W = weight.float()

    # Step 1: Geometry-constrained matrix
    W_geo = build_w_geo(W, rank, sparsity_ratio)

    # Step 2: SVD on W_Geo
    U_geo, S_geo, Vh_geo = torch.linalg.svd(W_geo, full_matrices=False)

    # Step 3: Top-r components (Equations 2 & 3)
    S_r = S_geo[:rank]
    S_sqrt = torch.sqrt(S_r)  # Σ_Geo[:r,:r]^{1/2}

    # A_Geo = Σ^{1/2}_{Geo[:r,:r]} V^T_{Geo[:,:r]}   → shape [r, in]
    A_geo = torch.diag(S_sqrt) @ Vh_geo[:rank, :]

    # B_Geo = U_{Geo[:,:r]} Σ^{1/2}_{Geo[:r,:r]}      → shape [out, r]
    B_geo = U_geo[:, :rank] @ torch.diag(S_sqrt)

    # Step 4: Residual matrix (Equation 4)
    scaling = lora_alpha / rank
    W_res = W - scaling * (B_geo @ A_geo)

    return A_geo.contiguous(), B_geo.contiguous(), W_res.contiguous()


# ---------------------------------------------------------------------------
# Model-level application
# ---------------------------------------------------------------------------


def add_geora_initialized_lora(
    model,
    rank: int = 16,
    sparsity_ratio: float = 0.2,
    hyper_param_type: str = "LLM-Adapters",
    target_modules=None,
) -> torch.nn.Module:
    """
    Apply GeoRA to a HuggingFace CausalLM model.

    GeoRA replaces the base weight with W_res (frozen residual anchor) and
    initializes lora_A / lora_B from geometry-aware SVD of W_Geo.

    Args:
        model:            HuggingFace pre-trained model
        rank:             LoRA rank r (paper default: 16)
        sparsity_ratio:   ρ for geometric mask (paper default: 0.2)
        hyper_param_type: "LLM-Adapters" or "QLoRA"

    Returns:
        PEFT model with GeoRA-initialized adapters
    """
    # --- Hyperparameter presets ---
    if target_modules is None:
        if hyper_param_type == "LLM-Adapters":
            lora_alpha = rank
            lora_dropout = 0.05
            target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
        elif hyper_param_type == "QLoRA":
            lora_alpha = rank
            lora_dropout = 0.1
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            raise ValueError(f"Unknown hyper_param_type: {hyper_param_type}")

    # --- Build PEFT model with standard LoRA config ---
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=rank,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    logger.info(f"Starting GeoRA initialization...")
    logger.info(
        f"Rank={rank} | SparsityRatio(ρ)={sparsity_ratio} | Targets={target_modules}"
    )
    start_time = time.time()

    # --- SVD-based GeoRA initialization ---
    with torch.no_grad():
        for name, module in model.named_modules():
            # Target LoRA-wrapped Linear layers
            if (
                hasattr(module, "base_layer")
                and hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
            ):
                if type(module.base_layer).__name__ != "Linear":
                    continue
                if not any(proj in name for proj in target_modules):
                    continue

                try:
                    base_weight = module.base_layer.weight.data  # [out, in]

                    A_geo, B_geo, W_res = initialize_geora_layer(
                        base_weight.float(),
                        rank=rank,
                        lora_alpha=lora_alpha,
                        sparsity_ratio=sparsity_ratio,
                    )

                    device = base_weight.device
                    dtype = base_weight.dtype

                    # Replace base weight with frozen residual anchor W_res
                    module.base_layer.weight.data = W_res.to(device=device, dtype=dtype)

                    # Initialize adapters from geometry-constrained SVD
                    module.lora_A["default"].weight.data = A_geo.to(
                        device=device, dtype=dtype
                    )
                    module.lora_B["default"].weight.data = B_geo.to(
                        device=device, dtype=dtype
                    )

                    logger.info(f"  GeoRA initialized: {name}")
                except Exception as e:
                    logger.warning(f"  Skipped {name}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"GeoRA initialization completed in {elapsed:.2f}s")

    return model


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    model = add_geora_initialized_lora(
        model=model,
        rank=16,
        sparsity_ratio=0.2,
        hyper_param_type="LLM-Adapters",
    )

    model.print_trainable_parameters()
    print("GeoRA model ready for RLVR training.")
