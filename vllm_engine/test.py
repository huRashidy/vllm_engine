import sys
sys.path.append("..")  # Add the parent directory to the module search path
from block_allocator import BlockAllocator


if __name__ == "__main__":
    alloc_class = BlockAllocator(num_blocks=5, block_size=16)

    alloc_class.allocate("request_1", 20)
    alloc_class.allocate("request_2", 30)
    print("Current free blocks:", alloc_class.free_blocks)
    