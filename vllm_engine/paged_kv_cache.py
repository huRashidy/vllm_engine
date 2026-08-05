"""
Paged KV-Cache Module
=====================
Simulates scattered physical VRAM allocation for Key and Value tensors
and computes Scaled Dot-Product Attention using PagedAttention lookup.
"""

import math
from typing import List
import torch


class PagedKVCache:
    """
    Physical 5D Tensor Storage for Key-Value vectors in VRAM.
    Shape: (num_blocks, 2, block_size, num_heads, head_dim)
    """
    def __init__(
        self, 
        num_blocks: int, 
        block_size: int, 
        num_heads: int, 
        head_dim: int, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Dim 1: Index 0 = Key Cache, Index 1 = Value Cache
        self.kv_cache = torch.zeros(
            (num_blocks, 2, block_size, num_heads, head_dim),
            dtype=torch.float32,
            device=device
        )

    def write_kv(
        self, 
        physical_block_idx: int, 
        block_offset: int, 
        k: torch.Tensor, 
        v: torch.Tensor
    ):
        """Writes Key and Value vectors for a single token into physical VRAM."""
        self.kv_cache[physical_block_idx, 0, block_offset] = k
        self.kv_cache[physical_block_idx, 1, block_offset] = v

    def paged_attention_forward(
        self, 
        query: torch.Tensor, 
        block_table: List[int], 
        total_tokens: int
    ) -> torch.Tensor:
        """
        Translates logical token indices to scattered physical addresses
        and computes Scaled Dot-Product Attention without contiguous VRAM allocation.
        """
        keys, values = [], []
        
        # 1. Map logical sequence index -> physical VRAM block + offset
        for i in range(total_tokens):
            table_idx = i // self.block_size
            offset = i % self.block_size
            phys_block = block_table[table_idx]
            
            keys.append(self.kv_cache[phys_block, 0, offset])
            values.append(self.kv_cache[phys_block, 1, offset])

        # 2. Gather vectors into attention shape: (1, num_heads, total_tokens, head_dim)
        K = torch.stack(keys, dim=0).transpose(0, 1).unsqueeze(0)
        V = torch.stack(values, dim=0).transpose(0, 1).unsqueeze(0)
        
        Q = query.unsqueeze(0).unsqueeze(2).to(self.device)  # (1, num_heads, 1, head_dim)

        # 3. Scaled Dot-Product Attention computation
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        return output.squeeze(0).squeeze(1)