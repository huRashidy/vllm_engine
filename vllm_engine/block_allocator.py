"""
Block Allocator Module
======================
Manages virtual memory allocation for physical KV-Cache blocks stored in GPU VRAM.
Implements an Operating System-style Paging Model to eliminate internal fragmentation.
"""

from typing import List, Dict, Optional


class BlockAllocator:
    """
    Virtual Memory Allocator for PagedAttention KV-Cache blocks.
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}

    def get_num_required_blocks(self, num_tokens: int) -> int:
        """Calculates required physical blocks using integer ceiling division."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        """Allocates physical blocks on-demand for a new request."""
        needed_blocks = self.get_num_required_blocks(num_tokens)
        
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError(
                f"Out of Memory (OOM)! Requested {needed_blocks} blocks, "
                f"but only {len(self.free_blocks)} free blocks remain in VRAM."
            )

        allocated = [self.free_blocks.pop(0) for _ in range(needed_blocks)]
        self.block_tables[request_id] = allocated
        return allocated

    def append_slot(self, request_id: str, current_total_tokens: int) -> Optional[int]:
        """
        Allocates an additional physical block if token count expands past block boundary.
        """
        table = self.block_tables[request_id]
        if current_total_tokens > len(table) * self.block_size:
            if not self.free_blocks:
                return None  # Out of VRAM / Signal preemption required
            new_block = self.free_blocks.pop(0)
            table.append(new_block)
            return new_block
        return table[-1]

    def free(self, request_id: str):
        """Reclaims physical blocks when a request completes."""
        if request_id in self.block_tables:
            freed = self.block_tables.pop(request_id)
            self.free_blocks.extend(freed)

    def num_free_blocks(self) -> int:
        """Returns the number of currently available physical VRAM blocks."""
        return len(self.free_blocks)