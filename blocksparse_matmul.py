
import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    ],

    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_block_pointers_blocksparsity(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, b_mask_step_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_mask_bk, stride_mask_bn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    b_mask_step_n_ptr = b_mask_step_ptr + pid_n * stride_mask_bn
    num_b_steps = tl.load(b_mask_step_n_ptr)
    
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))
    
    #print(a_block_ptr)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_b_steps):
        
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        b_mask_step_n_ptr += stride_mask_bk
        b_step_n = tl.load(b_mask_step_n_ptr)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K * b_step_n))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K * b_step_n, 0))
        
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        
    c = accumulator.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def get_step_from_b_mask(mask):
    ret = torch.zeros((mask.shape[0]+1, mask.shape[1]), device=mask.device)
    for i in range(mask.shape[1]):
        pos_ind = mask[:, i].nonzero().squeeze()
        ret[0, i] = pos_ind.numel()
        if pos_ind.numel() > 1:
            step = pos_ind[1:] - pos_ind[:-1]
            ret[1, i] = pos_ind[0]
            ret[2:len(step)+2, i] = step
        elif pos_ind.numel() == 1:
            step = torch.tensor(pos_ind)
            ret[1, i] = pos_ind
        
    return ret.to(int)
    

# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a, b, mask_step):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel_with_block_pointers_blocksparsity[grid](
        a, b, c, mask_step,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        mask_step.stride(0), mask_step.stride(1)
    )
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((128, 128), device='cuda', dtype=torch.float16)
    b = torch.randn((128, 128), device='cuda', dtype=torch.float16)
    b_mask = torch.randint(0, 2, (128//32, 128//64), device='cuda').to(torch.float16)
    mask_step = get_step_from_b_mask(b_mask)
    b_mask_up = torch.nn.functional.interpolate(b_mask[None, None, ...], scale_factor=(32, 64)).squeeze().to(torch.int)
    triton_output = matmul(a, b, mask_step.to(torch.int))
    torch_output = torch.matmul(a, b * b_mask_up)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
