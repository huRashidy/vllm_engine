🚀 Mini-vLLM: Minimal PagedAttention & Continuous Batching Engine

A lightweight, zero-dependency PyTorch implementation of the core architectural features powering modern high-throughput LLM inference engines (such as vLLM).

This repository demonstrates how to eliminate KV-Cache memory fragmentation and maximize GPU hardware throughput using Operating System Paging Principles and Iteration-Level Dynamic Scheduling.

🏗️ System Architecture

Traditional LLM serving pre-allocates contiguous memory for maximum sequence lengths (e.g., 2048 tokens), leading to over 60%–80% internal memory fragmentation.

Mini-vLLM solves this by decoupling logical token sequences from physical VRAM allocations:

Logical Sequence (Request A): [ T0, T1, ... T15 | T16, T17, ... T31 | T32 ... ]
                                      |                 |
Block Table Mapping:            Block Table [0]   Block Table [1]
                                      |                 |
                                      v                 v
Physical VRAM Storage:         [ Physical Block 2 ] [ Physical Block 7 ] (Non-contiguous)


🔑 Key Engineering Concepts Implemented

1. Dynamic Virtual Memory Paging (BlockAllocator)

OS-Style Memory Management: Divides physical VRAM into fixed-size block slots (e.g., 16 tokens/block).

Zero Internal Fragmentation: Allocates physical memory blocks on-demand as sequences grow during the Decode phase.

Integer Ceiling Allocation Math:


$$\text{Required Blocks} = \left\lfloor \frac{N + \text{block\_size} - 1}{\text{block\_size}} \right\rfloor$$

2. Paged KV Cache & Addressing (PagedKVCache)

Scattered Physical Storage: Holds Key and Value tensors in a single unified 5D physical VRAM tensor:


$$\text{Shape: } (\text{num\_blocks}, 2, \text{block\_size}, \text{num\_heads}, \text{head\_dim})$$

Logical-to-Physical Translation: Maps any logical sequence token index $i$ to physical storage on-the-fly without copying VRAM data:


$$\text{Table Index} = i \mathbin{/\!/} \text{block\_size}, \quad \text{Block Offset} = i \bmod \text{block\_size}$$

3. Iteration-Level Continuous Batching (ContinuousScheduler)

Dynamic Batching: Replaces static batching with token-level scheduling.

Immediate Eviction: Finished requests drop out immediately at step $t$, instantly freeing physical blocks for waiting requests at step $t+1$.

Prefill / Decode Separation: Simultaneously manages compute-bound Prefill requests (prompt processing) and memory-bound Decode requests (token generation).

📂 Project Structure

.
├── main.py                     # Benchmark runner and execution simulation
└── vllm_engine/
    ├── __init__.py
    ├── block_allocator.py      # Physical block allocation & VRAM management
    ├── paged_kv_cache.py       # Physical 5D VRAM tensor & PagedAttention calculation
    └── scheduler.py            # Iteration-level continuous batching scheduler


🚀 Quickstart

Prerequisites

Python 3.8+

torch (PyTorch)

Running the Engine Benchmark

Clone the repository:

git clone https://github.com/your-username/mini-vllm-engine.git
cd mini-vllm-engine


Run the simulation:

python main.py


📊 Sample Execution Output

=================================================================
🚀 Initializing Mini-vLLM Engine Benchmark Simulation
=================================================================
Initial Free VRAM Blocks: 8 / 8

--- Iteration Step 1 ---
  📥 [INJECTED] Req_A joined active batch (Prompt: 20 tokens).
  📥 [INJECTED] Req_B joined active batch (Prompt: 30 tokens).
  ⚡ [PREFILL] Processing prompt for Req_A...
  ⚡ [PREFILL] Processing prompt for Req_B...
  Free Memory Blocks Remaining: 4 / 8

--- Iteration Step 2 ---
  🔤 [DECODE] Req_A generated token 1/2 (Attention Output: torch.Size([4, 32]))
  🔤 [DECODE] Req_B generated token 1/3 (Attention Output: torch.Size([4, 32]))
  Free Memory Blocks Remaining: 4 / 8

--- Iteration Step 3 ---
  🎉 [COMPLETED] Req_A finished -> Freed physical VRAM blocks.
  📥 [INJECTED] Req_C joined active batch (Prompt: 15 tokens).
  ⚡ [PREFILL] Processing prompt for Req_C...
  🔤 [DECODE] Req_B generated token 2/3 (Attention Output: torch.Size([4, 32]))
  Free Memory Blocks Remaining: 5 / 8

✅ Mini-vLLM Engine Simulation Completed Successfully!


📚 References & Further Reading

vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)

vLLM Official Documentation