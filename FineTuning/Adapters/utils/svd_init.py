"""
svd_init.py
-----------
SVD utilities for parameter-efficient adapter initialization.

Handles SVD decomposition of weight matrices, caching, and initialization
of frozen low-rank matrices for adapters like LoRA-XS.

This module is designed to be reusable across different adapter types:
- LoRA-XS: uses SVD-computed A/B matrices as frozen components
- Custom RxR: can use SVD for initialization or direct weight decomposition
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
import hashlib

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


def _get_cache_dir() -> Path:
    """Get or create cache directory for SVD decompositions."""
    cache_dir = Path.home() / ".cache" / "adapter_svd"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# fast, practically unique for pretrained weights
def _get_weight_hash(weight: np.ndarray) -> str:
    # sample a few statistics instead of hashing all bytes
    key = f"{weight.shape}_{weight.mean():.6f}_{weight.std():.6f}_{weight[0,0]:.6f}_{weight[-1,-1]:.6f}"
    return hashlib.md5(key.encode()).hexdigest()

def _get_cache_path(weight_hash: str, rank: int, n_iter: int) -> Path:
    """Construct cache file path for SVD decomposition."""
    cache_dir = _get_cache_dir()
    filename = f"svd_{weight_hash}_r{rank}_iter{n_iter}.pkl"
    return cache_dir / filename


def compute_svd(
    weight: np.ndarray,
    rank: int,
    n_iter: int = 30,
    use_cache: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute truncated SVD decomposition of a weight matrix.
    
    Uses scipy.sparse.linalg.svds for memory-efficient computation of large matrices.
    Optionally caches results to avoid recomputation.
    
    Args:
        weight: Weight matrix of shape (m, n). For transformers, typically (out_features, in_features).
        rank: Number of singular vectors to compute (the 'r' in LoRA-XS).
        n_iter: Number of iterations for Lanczos algorithm (higher = better accuracy, slower).
        use_cache: If True, cache computed decompositions and reuse if seen before.
        device: Device to move tensors to after computation (e.g., "cuda", "cpu").
    
    Returns:
        (U, S, Vt): SVD decomposition where weight ≈ U @ diag(S) @ Vt
            - U: shape (m, rank), left singular vectors
            - S: shape (rank,), singular values
            - Vt: shape (rank, n), right singular vectors (transposed)
    
    Note:
        - scipy.sparse.linalg.svds returns (U, S, Vt) directly
        - We return (U, S, Vt) where weight ≈ U @ diag(S) @ Vt
        - For LoRA-XS: typically use U as frozen_A and Vt as frozen_B
    """
    
    # Check cache first
    if use_cache:
        weight_hash = _get_weight_hash(weight)
        cache_path = _get_cache_path(weight_hash, rank, n_iter)
        
        if cache_path.exists():
            logger.info(f"Loading cached SVD from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return cached['U'], cached['S'], cached['Vt']
    
    # Compute SVD
    logger.info(f"Computing SVD: weight shape {weight.shape}, rank {rank}, n_iter {n_iter}")
    
    # Handle the case where rank >= min(weight.shape) — use full SVD
    if rank >= min(weight.shape):
        logger.warning(f"Requested rank {rank} >= min(shape) {min(weight.shape)}, using full SVD")
        U, S, Vt = np.linalg.svd(weight, full_matrices=False)
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    else:
        # Use truncated SVD for efficiency
        try:
            svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=42)
            svd.fit(weight)
            # We want U, S, Vt. TruncatedSVD computes them intrinsically but doesn't easily expose U.
            # svd.components_ is Vt, svd.singular_values_ is S.
            Vt = svd.components_
            S = svd.singular_values_
            # U = transform(weight) @ inv(diag(S))
            transformed = svd.transform(weight)
            # prevent division by zero
            safe_S = np.where(S == 0, 1e-10, S)
            U = transformed / safe_S
        except Exception as e:
            logger.warning(f"TruncatedSVD failed: {e}, falling back to full SVD")
            U_full, S_full, Vt_full = np.linalg.svd(weight, full_matrices=False)
            U, S, Vt = U_full[:, :rank], S_full[:rank], Vt_full[:rank, :]
    
    # Cache result
    if use_cache:
        weight_hash = _get_weight_hash(weight)
        cache_path = _get_cache_path(weight_hash, rank, n_iter)
        
        with open(cache_path, 'wb') as f:
            pickle.dump({'U': U, 'S': S, 'Vt': Vt}, f)
        logger.info(f"Cached SVD to {cache_path}")
    
    return U, S, Vt


def get_frozen_ab_matrices_major(
    weight: np.ndarray,
    rank: int,
    n_iter: int = 30,
    use_cache: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract frozen A and B matrices from SVD decomposition for LoRA-XS.
    
    In LoRA-XS:
    - weight ≈ A @ R @ B  (where R is trainable, A and B are frozen)
    - A and B are derived from SVD: A = U @ S, B = Vt
    
    Args:
        weight: Weight matrix (typically from linear layer).
        rank: Rank for SVD.
        n_iter: Lanczos iterations.
        use_cache: Whether to cache SVD results.
        device: Device for output tensors.
    
    Returns:
        (A, B) as torch.Tensors ready for use in LoRA-XS initialization.
    """
    
    U, S, Vt = compute_svd(weight, rank, n_iter, use_cache, device="cpu")
    
    # Match TruncatedSVD transform: A = U * S, B = Vt
    A = (U * S[np.newaxis, :]).astype(weight.dtype)
    B = Vt.astype(weight.dtype)
    
    # Convert to torch tensors and move to device
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
    B_tensor = torch.tensor(B, dtype=torch.float32).to(device)
    
    return A_tensor, B_tensor


def get_frozen_ab_matrices_minor(
    weight: np.ndarray,
    rank: int,
    n_iter: int = 30,
    use_cache: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract frozen A and B matrices from the MINOR singular components of W.
    
    Mirrors get_frozen_ab_matrices exactly but uses the smallest singular values/vectors
    instead of the largest — following MiLoRA's insight that minor components correspond
    to less-optimized subspace, reducing interference with pretrained knowledge.

    Output convention is identical to get_frozen_ab_matrices:
        A: (out_features, rank) — scaled by singular values, maps to lora_B
        B: (rank, in_features) — unscaled right singular vectors, maps to lora_A
    """
    weight_clean = weight.astype(np.float32)

    # Check cache
    if use_cache:
        weight_hash = _get_weight_hash(weight_clean)
        cache_path = _get_cache_path(f"{weight_hash}_minor", rank, n_iter)
        if cache_path.exists():
            logger.info(f"Loading cached minor SVD from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            U, S, Vt = cached["U"], cached["S"], cached["Vt"]
        else:
            U, S, Vt = _compute_minor_svd(weight_clean, rank)
            with open(cache_path, "wb") as f:
                pickle.dump({"U": U, "S": S, "Vt": Vt}, f)
            logger.info(f"Cached minor SVD to {cache_path}")
    else:
        U, S, Vt = _compute_minor_svd(weight_clean, rank)

    # Match major convention: A = U * S (scaled), B = Vt (unscaled)
    A = (U * S[np.newaxis, :]).astype(weight.dtype)
    B = Vt.astype(weight.dtype)

    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
    B_tensor = torch.tensor(B, dtype=torch.float32).to(device)

    return A_tensor, B_tensor


def _compute_minor_svd(
    weight: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the minor (smallest magnitude) singular components of a weight matrix.
    Uses scipy.sparse.linalg.svds with which='SM' for efficiency.
    Falls back to full SVD and tail-slicing if svds fails.
    """
    from scipy.sparse.linalg import svds

    try:
        # 'SM' = smallest magnitude — scipy uses shift-invert internally, fast
        U, S, Vt = svds(weight, k=rank, which="SM")

        # svds returns ascending order for SM — flip so smallest is index 0
        # (mirrors descending convention of the major case)
        U = np.flip(U, axis=1).copy()
        S = np.flip(S).copy()
        Vt = np.flip(Vt, axis=0).copy()

    except Exception as e:
        logger.warning(f"Minor svds failed: {e}, falling back to full SVD tail")
        U_full, S_full, Vt_full = np.linalg.svd(weight, full_matrices=False)
        # Take the tail (smallest) components
        U = U_full[:, -rank:][:, ::-1].copy()
        S = S_full[-rank:][::-1].copy()
        Vt = Vt_full[-rank:, :][::-1, :].copy()

    return U, S, Vt

def get_adaptive_frozen_ab_matrices_major(
    weight: np.ndarray,
    max_rank: int,
    threshold: float = 0.90,
    min_rank: int = 2,
    n_iter: int = 30,
    use_cache: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extract frozen A and B matrices using an adaptively chosen rank.
    
    Computes SVD with `max_rank`, then dynamically selects rank `k` based on
    the cumulative singular value threshold.
    
    Returns:
        (A, B, k) where A and B are sized according to the selected rank k.
    """
    U, S, Vt = compute_svd(weight, max_rank, n_iter, use_cache, device="cpu")
    
    # Convert singular values to a PyTorch tensor
    S_tensor = torch.tensor(S)
    
    # 1. Square the singular values to get energy / variance
    S_squared = S_tensor ** 2
    
    # 2. Calculate cumulative energy ratio
    # (Scipy's 'SM' returns values sorted from smallest to largest, 
    # so cumulative sum naturally accumulates minor component energy first)
    cumulative_energy = torch.cumsum(S_squared, dim=0) / S_squared.sum()
    
    # 3. Find the rank k that captures the target threshold of minor energy
    k = (cumulative_energy < threshold).sum().item() + 1
    
    # 4. Enforce your hard minimum and maximum boundary constraints
    k = max(min_rank, min(k, max_rank))
    
    # Truncate U, S, Vt to k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Compute A and B based on k
    A = (U_k * S_k[np.newaxis, :]).astype(weight.dtype)
    B = Vt_k.astype(weight.dtype)
    
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)
    B_tensor = torch.tensor(B, dtype=torch.float32).to(device)
    
    return A_tensor, B_tensor, k


def verify_svd_reconstruction(
    weight: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
) -> Dict[str, float]:
    """
    Verify SVD reconstruction quality (for debugging/logging).
    
    Args:
        weight: Original weight matrix.
        U, S, Vt: SVD components.
    
    Returns:
        Dict with metrics:
            - "frobenius_norm": ||weight - reconstructed||_F / ||weight||_F
            - "spectral_norm": ||weight - reconstructed||_2 / ||weight||_2
            - "max_relative_error": max(|weight - reconstructed|) / max(|weight|)
    """
    
    # Reconstruct weight from SVD components
    reconstructed = U @ np.diag(S) @ Vt
    
    # Compute error metrics
    error = weight - reconstructed
    
    frobenius_error = np.linalg.norm(error, 'fro') / np.linalg.norm(weight, 'fro')
    spectral_error = np.linalg.norm(error, 2) / np.linalg.norm(weight, 2)
    max_relative_error = np.max(np.abs(error)) / np.max(np.abs(weight))
    
    return {
        "frobenius_norm": float(frobenius_error),
        "spectral_norm": float(spectral_error),
        "max_relative_error": float(max_relative_error),
    }


def clear_svd_cache() -> None:
    """Clear all cached SVD decompositions."""
    cache_dir = _get_cache_dir()
    count = 0
    for cache_file in cache_dir.glob("svd_*.pkl"):
        cache_file.unlink()
        count += 1
    logger.info(f"Cleared {count} cached SVD decompositions from {cache_dir}")
