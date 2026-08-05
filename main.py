"""
Mini-vLLM: Minimal PagedAttention & Continuous Batching Inference Engine
========================================================================

Architecture & Memory Layout:
----------------------------
  Logical Tokens (Request 1): [ T0, T1, ... T15 | T16, T17, ... T31 | T32 ... ]
                                      |                 |
  Logical-to-Physical Map:      Block Table [0]   Block Table [1]
                                      |                 |
                                      v                 v
  Physical VRAM Storage:       [ Physical Block 5 ] [ Physical Block 12 ] (Non-contiguous)

Key Engineering Features Implemented:
1. BlockAllocator: Dynamic virtual memory management for KV-Cache (Zero internal fragmentation).
2. PagedKVCache: Physical 5D VRAM tensor holding Key and Value tensors in scattered blocks.
3. PagedAttention Indexing: Dynamic logical token index to 3D physical address translation.
4. ContinuousScheduler: Iteration-level scheduling (dynamic prefill/decode batching & preemption safety).
"""

"""
Mini-vLLM Engine Benchmark Entrypoint
====================================
Runs an end-to-end continuous batching simulation integrating BlockAllocator,
PagedKVCache, and ContinuousScheduler.
"""

import torch
from vllm_engine.block_allocator import BlockAllocator
from vllm_engine.paged_kv_cache import PagedKVCache
from vllm_engine.scheduler import Request, ContinuousScheduler


def run_benchmark():
    print("=" * 65)
    print("🚀 Initializing Mini-vLLM Engine Benchmark Simulation")
    print("=" * 65)

    BLOCK_SIZE = 16
    NUM_BLOCKS = 8
    NUM_HEADS = 4
    HEAD_DIM = 32

    # Initialize Engine Subsystems
    allocator = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    kv_cache = PagedKVCache(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    scheduler = ContinuousScheduler(max_batch_size=2, allocator=allocator)

    # Add Test Requests
    scheduler.add_request(Request("Req_A", prompt_len=20, max_gen_len=2))
    scheduler.add_request(Request("Req_B", prompt_len=30, max_gen_len=3))
    scheduler.add_request(Request("Req_C", prompt_len=15, max_gen_len=2))

    print(f"Initial Free VRAM Blocks: {allocator.num_free_blocks()} / {NUM_BLOCKS}\n")

    for step in range(1, 7):
        print(f"--- Iteration Step {step} ---")
        prefills, decodes = scheduler.schedule_iteration()

        # Handle Prefill Phase
        for req in prefills:
            print(f"  ⚡ [PREFILL] Processing prompt for {req.req_id}...")
            bt = allocator.block_tables[req.req_id]
            for tok_idx in range(req.prompt_len):
                p_block = bt[tok_idx // BLOCK_SIZE]
                p_offset = tok_idx % BLOCK_SIZE
                dummy_k = torch.randn(NUM_HEADS, HEAD_DIM)
                dummy_v = torch.randn(NUM_HEADS, HEAD_DIM)
                kv_cache.write_kv(p_block, p_offset, dummy_k, dummy_v)
            req.is_prefilled = True

        # Handle Decode Phase with PagedAttention
        for req in decodes:
            req.generated_tokens += 1
            allocator.append_slot(req.req_id, req.total_tokens)
            
            bt = allocator.block_tables[req.req_id]
            dummy_q = torch.randn(NUM_HEADS, HEAD_DIM)
            attn_out = kv_cache.paged_attention_forward(dummy_q, bt, req.total_tokens - 1)
            
            print(f"  🔤 [DECODE] {req.req_id} generated token {req.generated_tokens}/{req.max_gen_len} "
                  f"(Attention Output: {attn_out.shape})")

        print(f"  Free Memory Blocks Remaining: {allocator.num_free_blocks()} / {NUM_BLOCKS}\n")

    print("✅ Mini-vLLM Engine Simulation Completed Successfully!")


if __name__ == "__main__":
    run_benchmark()