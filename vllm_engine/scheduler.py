"""
Continuous Batching Scheduler Module
====================================
Manages iteration-level request lifecycle state, dynamic prefill/decode batching,
and VRAM memory safety.
"""

from typing import List, Tuple
from vllm_engine.block_allocator import BlockAllocator


class Request:
    """Tracks individual request sequence generation lifecycle state."""
    def __init__(self, req_id: str, prompt_len: int, max_gen_len: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.generated_tokens = 0
        self.is_prefilled = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_gen_len


class ContinuousScheduler:
    """
    Iteration-Level Continuous Batching Scheduler.
    """
    def __init__(self, max_batch_size: int, allocator: BlockAllocator):
        self.max_batch_size = max_batch_size
        self.allocator = allocator
        self.waiting_queue: List[Request] = []
        self.running_batch: List[Request] = []

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def schedule_iteration(self) -> Tuple[List[Request], List[Request]]:
        """
        Executes one iteration step of dynamic batch scheduling:
        1. Evicts finished requests and reclaims VRAM blocks.
        2. Injects waiting requests if memory capacity allows.
        3. Classifies batch into prefill vs decode requests.
        """
        # Step 1: Evict finished requests
        finished = [r for r in self.running_batch if r.is_finished]
        for r in finished:
            self.allocator.free(r.req_id)
            self.running_batch.remove(r)
            print(f"  🎉 [COMPLETED] {r.req_id} finished -> Freed physical VRAM blocks.")

        # Step 2: Inject waiting requests if VRAM budget allows
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            candidate = self.waiting_queue[0]
            req_blocks = self.allocator.get_num_required_blocks(candidate.prompt_len)
            
            if self.allocator.num_free_blocks() >= req_blocks:
                req = self.waiting_queue.pop(0)
                self.allocator.allocate(req.req_id, req.prompt_len)
                self.running_batch.append(req)
                print(f"  📥 [INJECTED] {req.req_id} joined active batch (Prompt: {req.prompt_len} tokens).")
            else:
                break  # Stop injecting until active requests free space

        # Step 3: Classify requests into Prefill vs Decode phases
        prefills = [r for r in self.running_batch if not r.is_prefilled]
        decodes = [r for r in self.running_batch if r.is_prefilled]

        return prefills, decodes