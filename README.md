# TritonBlockSparseMatmul

A matmul kernel with block-sparsity based on [OpenAI-Triton's Matmul](https://github.com/openai/triton/blob/main/python/tutorials/08-experimental-block-pointer.py)) that works in Pytorch.

Currently, it's required to build triton-2.1.0 from source to use the newest block pointer. 

Benchmarking 
---
Below is a comparison with Pytorch native cublas-based matmul on the throughput on A100, when `BLOCK_SIZE_M=128`, `BLOCK_SIZE_K=32` and `BLOCK_SIZE_N=64`. 
<p align="center">
  <img src="benchmark.png" />
</p>
This implementation is faster than pytorch on >50% block-sparsity, which improves from the [huggingface's implementation](https://github.com/huggingface/pytorch_block_sparse)

Related Work
---
HuggingFace implemented a [blocksparse gemm kernel earlier](https://github.com/huggingface/pytorch_block_sparse) based on CUTLASS, but unfortunately the speedup isn't satisfactory yet for 50% sparsity. 

OpenAI also implemented [one for tensorflow](https://github.com/openai/blocksparse), Pytorch support is unfortunately not available. 
