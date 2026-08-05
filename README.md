# 🚀 Mini-vLLM Engine  
### Minimal PagedAttention + Continuous Batching in Pure PyTorch

A lightweight, zero-dependency PyTorch implementation of the core architectural ideas behind modern high-throughput LLM serving systems (inspired by [vLLM](https://github.com/vllm-project/vllm)).

---

## ✨ Why this project?

Traditional LLM inference engines often pre-allocate large contiguous KV-cache regions (e.g., for 2048+ tokens per request), causing substantial internal memory waste.

**Mini-vLLM** demonstrates a cleaner approach:

- 🧠 **Paged KV cache** (logical tokens decoupled from physical VRAM)
- 📦 **Block-based allocation** (OS-style paging principles)
- 🔄 **Continuous batching** (iteration-level dynamic scheduling)

This helps reduce memory fragmentation and improve effective GPU throughput.

---

## 🏗️ Architecture at a glance

Mini-vLLM separates **logical sequence growth** from **physical memory layout**.

```text
Logical Sequence (Req A):
[T0..T15] [T16..T31] [T32..]

         │        │
         ▼        ▼
Block Table:
   [0]      [1]   ...

         │        │
         ▼        ▼
Physical VRAM Blocks:
 [Block 2] [Block 7] ... (non-contiguous)
```

---

## 🔑 Core concepts implemented

### 1) Dynamic Virtual Memory Paging (`BlockAllocator`)

- Divides VRAM into fixed token blocks (e.g., `block_size = 16`)
- Allocates blocks on-demand as decode progresses
- Avoids large up-front contiguous reservations

\[
\text{Required Blocks} = \left\lfloor \frac{N + \text{block\_size} - 1}{\text{block\_size}} \right\rfloor
\]

---

### 2) Paged KV Cache (`PagedKVCache`)

- Stores all K/V states in one 5D physical tensor:

\[
(\text{num\_blocks}, 2, \text{block\_size}, \text{num\_heads}, \text{head\_dim})
\]

- Resolves logical token index \(i\) → physical block location at runtime:

\[
\text{table\_index} = i // \text{block\_size}, \quad
\text{block\_offset} = i \bmod \text{block\_size}
\]

---

### 3) Iteration-Level Continuous Batching (`ContinuousScheduler`)

- Token-step scheduling instead of static batch execution
- Immediate eviction of completed requests
- Frees blocks instantly for waiting requests on next iteration
- Handles both:
  - **Prefill** (prompt processing; compute-heavy)
  - **Decode** (token generation; memory-heavy)

---

## 📂 Project structure

```text
.
├── main.py                     # Benchmark runner + execution simulation
└── vllm_engine/
    ├── __init__.py
    ├── block_allocator.py      # Physical block allocation / free list management
    ├── paged_kv_cache.py       # Paged KV tensor + address translation + attention
    └── scheduler.py            # Continuous batching scheduler
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.8+
- PyTorch (`torch`)

### Installation

```bash
git clone https://github.com/huRashidy/vllm_engine.git
cd vllm_engine
```

### Run benchmark simulation

```bash
python main.py
```

---

## 📊 Example output

```text
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
```

---

## 🎯 Learning goals

This project is useful if you want to understand:

- How PagedAttention removes KV-cache fragmentation
- How block tables map logical tokens to non-contiguous physical memory
- How continuous batching improves serving utilization
- Why prefill/decode phases have different system bottlenecks

---

## 📚 References

- **vLLM Paper (2023):**  
  *Efficient Memory Management for Large Language Model Serving with PagedAttention*  
  Kwon et al. (arXiv)
- **vLLM Project:** https://github.com/vllm-project/vllm
- **vLLM Documentation:** https://docs.vllm.ai/
